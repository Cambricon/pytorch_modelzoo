from __future__ import print_function
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../tools/utils/")
sys.path.append(cur_dir + "/models")

from metric import MetricCollector
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot, VOCDetectionResult
from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_300,VOC_512

import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.timer import Timer

parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-m', '--trained_model', default='weights/RFB300_80_5.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='./eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--cpu', default=False, type=bool,
                    help='Use cpu nms')
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
parser.add_argument('--device', default='mlu',
                    help='Use mlu or cuda to train model')
parser.add_argument('--precheckin', action='store_true', default=False,
                    help='under precheckin mode, we will only run two iterations')
parser.add_argument('--debug', action='store_true', default=False,
                    help='debugging mode')
args = parser.parse_args()

if args.debug:
    args.cpu = True

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

_USE_MLU = False
_USE_GPU = False
if args.device == 'mlu':
    import torch_mlu.core.mlu_model as ct
    _USE_MLU=True
elif args.device == 'cuda':
    if torch.cuda.is_available():
        _USE_GPU = True


_USE_GPU_NMS = False
_USE_MLU_NMS = False
_USE_CPU_NMS = False

if _USE_GPU and not args.cpu:
    from utils.nms_wrapper import nms
    _USE_GPU_NMS = True
elif _USE_MLU and not args.cpu:
    from torchvision.ops import nms
    _USE_MLU_NMS = True
else:
    from utils.nms_wrapper import nms
    _USE_CPU_NMS = True

class TestSetWrapper:
    def __init__(self, dataset_type, version, size):
        self.dataset_type = dataset_type
        self.img_dim = (300,512)[size=='512']
        self.rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[args.version == 'RFB_mobile']
        self.top_k = 200
        #p = (0.6,0.2)[version == 'RFB_mobile']
        self.num_classes = (21, 81)[dataset_type == 'COCO']

        if dataset_type == 'VOC':
            self.cfg = (VOC_300, VOC_512)[args.size == '512']
            self.testset = VOCDetection(VOCroot, VOCDetectionResult, [('2007', 'test')], None, AnnotationTransform())
            self.num_images = len(self.testset)
        else:
            self.testset = None
            print('Only VOC and COCO are supported now!')



def load_net(net, relative_data_path):
    print("trained_model=", args.trained_model)
    checkpoint = torch.load(args.trained_model, map_location='cpu')
    #state_dict = torch.load(relative_data_path)

    checkpoint_format = 0 # the original saving method
    if ("checkpoint_format" in checkpoint):
        print("checkpoint_format=", checkpoint['checkpoint_format'])
        checkpoint_format = checkpoint['checkpoint_format']
    else:
        print("no checkpoint_format")
    # create new OrderedDict that does not contain `module.`

    if checkpoint_format == 1:
        model_state_dict = checkpoint['model']
    else: # checkpoint_format == 0
        model_state_dict = checkpoint


    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    print('Finished loading model!')
    #print(net)
    return net


def test_net(save_folder, testset_wrapper, net, priors, detector, transform, max_per_image=300, thresh=0.005):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    testset = testset_wrapper.testset
    # dump predictions and assoc. ground truth to text file for now
    all_boxes = [[[] for _ in range(testset_wrapper.num_images)]
                 for _ in range(testset_wrapper.num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return

    metric_collector = MetricCollector(
        enable_only_benchmark=True,
        record_elapsed_time=True,
        record_hardware_time=True if args.device == 'mlu' else False)
    metric_collector.place()
    
    for i in range(testset_wrapper.num_images):
        img = testset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])

        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            if _USE_MLU:
                x = x.to(ct.mlu_device())
                scale = scale.to(ct.mlu_device())
            elif _USE_GPU:
                x = x.cuda()
                scale = scale.cuda()

        _t['im_detect'].tic()
        out = net(x)      # forward pass
        boxes, scores = detector.forward(out,priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores=scores[0]

        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image

        _t['misc'].tic()

        for j in range(1, testset_wrapper.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            if _USE_MLU_NMS:
                keep = nms(torch.from_numpy(c_bboxes).to('mlu'), torch.from_numpy(c_scores).to('mlu'), 0.45)
                c_dets = c_dets[keep.cpu().numpy(), :]
            elif _USE_GPU_NMS:
                keep = nms(c_dets, 0.45, force_cpu=False)
                c_dets = c_dets[keep, :]
            else:
                keep = nms(c_dets, 0.45, force_cpu=True)
                c_dets = c_dets[keep, :]

            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,testset_wrapper.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, testset_wrapper.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()
        #break # to be deleted

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                .format(i + 1, testset_wrapper.num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

        # MetricCollector record
        metric_collector.record()
        metric_collector.place()
        
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
        
    metric_collector.insert_metrics(
        net = "RFBNet",
        batch_size = 1,
        precision = "FP32",
        cards = 1,
        DPF_mode = "single")
    metric_collector.dump()
        
    print('Evaluating detections')
    testset.evaluate_detections(all_boxes, save_folder)


def main_worker(local_rank, args):
    if args.version == 'RFB_vgg':
        from models.RFB_Net_vgg import build_net
    elif args.version == 'RFB_E_vgg':
        from models.RFB_Net_E_vgg import build_net
    elif args.version == 'RFB_mobile':
        from models.RFB_Net_mobile import build_net
    else:
        print('Unkown version!')
        return

    testset_wrapper = TestSetWrapper(args.dataset, args.version, args.size)
    if testset_wrapper == None:
        return

    net = build_net('test', testset_wrapper.img_dim, testset_wrapper.num_classes)    # initialize detector
    net = load_net(net, args.trained_model)
    net = adapt_net(net) # to mlu first, then DDP

    priorbox = PriorBox(testset_wrapper.cfg)
    with torch.no_grad():
        priors = priorbox.forward()
        priors = adapt_priors(priors)

    detector = Detect(testset_wrapper.num_classes, 0, testset_wrapper.cfg)
    save_folder = os.path.join(args.save_folder, args.dataset)
    #print('args.save_folder = ', args.save_folder)
    #print('args.dataset = ', args.dataset)
    #print('net.size = ', net.size)
    test_net(save_folder, testset_wrapper, net, priors, detector,
             BaseTransform(net.size, testset_wrapper.rgb_means, (2, 0, 1)),
             testset_wrapper.top_k, thresh=0.01)
    
def adapt_priors(priors):
    if _USE_MLU:
        priors = priors.to(ct.mlu_device())
    elif _USE_GPU:
        priors = priors.cuda()
    return priors

def adapt_net(net):
    if _USE_MLU:
        net.to(torch.device('mlu'))
    elif _USE_GPU:
        net.cuda() # because we had previously set ct.set_device(dev_id),
                   # these is no need to specify the device id again,
        cudnn.benchmark = True
    return net

if __name__ == '__main__':
    main_worker(0, args)
