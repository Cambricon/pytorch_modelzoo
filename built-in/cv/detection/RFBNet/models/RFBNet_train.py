from __future__ import print_function
import sys
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../tools/utils/")
sys.path.append(cur_dir + "/models")
from metric import MetricCollector

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import VOCroot, COCOroot, VOCDetectionResult, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from layers.modules import MultiBoxLoss
from layers.functions import PriorBox
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument('--basenet', default='./weights/vgg16_reducedfc.pth',
                        help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=32,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=16,
                    type=int, help='Number ofnnnn// workers used in dataloading')
parser.add_argument('--device', default='cpu',
                    help='Use mlu or cuda to train model')
parser.add_argument('--nprocessor', default=1, type=int, help='# of processors')
parser.add_argument('--lr', '--learning-rate',
                    default=4e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume', action='store_true', help='resume indicator')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')
parser.add_argument('--resume_path', default='',
                        help='pretrained base model')
parser.add_argument('--unit_in_iters', action='store_true', default=False,
                    help='if True, rounds of trainings will be governed in terms of iterations instead of epoches')
parser.add_argument('-max','--max_epoch_or_iter', default=2,
                    type=int, help='epoches or iterations to go through for training')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='',
                    help='Location to save checkpoint models')
parser.add_argument('--node_rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--mode', default=0, type=str,
                    help='mode')
parser.add_argument('--cnmix', action='store_true', default=False,
        help='use cnmix for mixed precision training')
parser.add_argument('--opt_level', type=str, default='O0',
        help='choose level of mixing precision')
parser.add_argument('--pyamp', action='store_true', default=False,
                    help='use pytorch amp for mixed precision training')
parser.add_argument('--debug', action='store_true', default=True,
                    help='debugging mode')
parser.add_argument("--seed", default=-1, type=int)

args = parser.parse_args()

if args.debug:
    args.seed = 1

_USE_MLU = False
_USE_GPU = False

if args.device == 'mlu':
    import torch_mlu.core.mlu_model as ct
    _USE_MLU=True
    if args.cnmix:
        import cnmix
elif args.device == 'cuda':
    if torch.cuda.is_available():
        _USE_GPU = True
scaler = None
if args.pyamp:
    scaler = GradScaler()

if args.save_folder == "":
    args.save_folder = "weight_output_"
    if args.device == 'cpu' or args.device == 'cuda':
        args.save_folder += args.device
    else: #mlu
        if args.cnmix:
            args.save_folder +='290'
        else:
            args.save_folder +='370'
        if args.mode != '':
            args.save_folder += '_' + args.mode

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

print('args.cnmix=', args.cnmix)

if args.seed != -1:
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main_worker(local_rank, nprocessor_per_node, args):
    if args.version == 'RFB_vgg':
        from models.RFB_Net_vgg import build_net
    elif args.version == 'RFB_E_vgg':
        from models.RFB_Net_E_vgg import build_net
    elif args.version == 'RFB_mobile':
        from models.RFB_Net_mobile import build_net
    else:
        print('Unkown version!')

    world_size = args.world_size * nprocessor_per_node
    gloabal_rank = args.node_rank * nprocessor_per_node + local_rank

    dataset_wrapper = DataSetWrapper(args.dataset, args.version, args.size)
    if dataset_wrapper == None:
        return

    training_strategy = TrainingStrategy(dataset_wrapper.dataset_type, dataset_wrapper.dataset.name, dataset_wrapper.data_length,
                                        args.resume_epoch, args.max_epoch_or_iter, args.batch_size, world_size, args.num_workers)

    net_dict, optimizer_dict, mixed_dict = load_checkpoint(args.resume)

    net = build_net('train', dataset_wrapper.img_dim, dataset_wrapper.num_classes)
    net = load_net(net, net_dict, args.resume)
    #print(net)

    if args.distributed:
        dev_id = int(local_rank)
        if _USE_MLU:
            backend = "cncl"
            ct.set_device(dev_id)
        else:
            backend = "nccl"
            torch.cuda.set_device(dev_id) # equivalent of os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
        dist.init_process_group(backend=backend, init_method="env://", rank=gloabal_rank, world_size=world_size)

    net = adapt_net(net) # to mlu first, then DDP

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=training_strategy.momentum, weight_decay=training_strategy.weight_decay)
    #optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha = 0.9, eps=1e-08,
    #                      momentum=args.momentum, weight_decay=args.weight_decay)
    if optimizer_dict is not None:
        optimizer.load_state_dict(optimizer_dict)
    optimizer = adapt_optimizer(optimizer)

    if args.cnmix:
        net, optimizer = cnmix.initialize(net, optimizer, opt_level=args.opt_level)
        cnmix.cnmix_set_amp_quantify_params('all', {'batch_size': training_strategy.batch_size_per_processor, 'data_num': dataset_wrapper.data_length})
        if mixed_dict is not None:
            cnmix.load_state_dict(mixed_dict)
    elif args.pyamp and scaler is not None and mixed_dict is not None:
        scaler.load_state_dict(mixed_dict)

    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[dev_id])

    criterion = MultiBoxLoss(dataset_wrapper.num_classes, 0.5, True, 0, True, 3, 0.5, False)
    criterion = adapt_criterion(criterion)

    priorbox = PriorBox(dataset_wrapper.cfg)
    with torch.no_grad():
        priors = priorbox.forward()
        priors = adapt_priors(priors)

    train(net, dataset_wrapper, training_strategy, optimizer, criterion, priors, args.distributed, local_rank, world_size)

    if args.distributed:
        dist.destroy_process_group()

def adapt_net(net):
    if _USE_MLU:
        net.to(torch.device('mlu'))
    elif _USE_GPU:
        net.cuda() # because we had previously set ct.set_device(dev_id),
                   # these is no need to specify the device id again,
        cudnn.benchmark = True
    return net

def adapt_optimizer(optimizer):
    if _USE_MLU:
        optimizer = ct.to(optimizer, torch.device("mlu"))
    return optimizer

def adapt_criterion(criterion):
    if _USE_MLU:
        criterion = criterion.to(ct.mlu_device())
    return criterion

def adapt_priors(priors):
    if _USE_MLU:
        priors = priors.to(ct.mlu_device())
    elif _USE_GPU:
        priors = priors.cuda()
    return priors

class DataSetWrapper:
    def __init__(self, dataset_type, version, size):
        self.dataset_type = dataset_type
        self.img_dim = (300,512)[size=='512']
        rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[args.version == 'RFB_mobile']
        p = (0.6,0.2)[version == 'RFB_mobile']
        self.num_classes = (21, 81)[dataset_type == 'COCO']

        if dataset_type == 'VOC':
            train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
            self.cfg = (VOC_300, VOC_512)[args.size == '512']
            self.dataset = VOCDetection(VOCroot, VOCDetectionResult, train_sets, preproc(self.img_dim, rgb_means, p), AnnotationTransform())
            self.data_length = len(self.dataset)
        elif dataset_type == 'COCO':
            train_sets = [('2014', 'train'),('2014', 'valminusminival')]
            self.cfg = (COCO_300, COCO_512)[args.size == '512']
            self.dataset = COCODetection(COCOroot, train_sets, preproc(
            self.img_dim, rgb_means, p))
            self.data_length = len(self.dataset)
        elif dataset_type == 'RFB_mobile':
            self.cfg = COCO_mobile_300
            self.dataset = None
            print('Only VOC and COCO are supported now!')
        else:
            self.dataset = None
            print('Only VOC and COCO are supported now!')

class TrainingStrategy:
    def __init__(self, dataset_type, dataset_name, dataset_len, resume_epoch, max_epoch_or_iter, batch_size, world_size, num_workers):
        self.weight_decay = args.weight_decay
        self.gamma = args.gamma
        self.momentum = args.momentum

        self.iterations_per_epoch = dataset_len // batch_size

        stepvalues_VOC = (150 * self.iterations_per_epoch, 200 * self.iterations_per_epoch, 250 * self.iterations_per_epoch)
        stepvalues_COCO = (90 * self.iterations_per_epoch, 120 * self.iterations_per_epoch, 140 * self.iterations_per_epoch)
        self.stepvalues = (stepvalues_VOC,stepvalues_COCO)[dataset_type=='COCO']
        print('Training',args.version, 'on', dataset_name)

        if resume_epoch > 0:
            self.start_iter = resume_epoch * self.iterations_per_epoch
        else:
            self.start_iter = 0
        if not args.unit_in_iters:
            self.end_iter = (resume_epoch + max_epoch_or_iter) * self.iterations_per_epoch
        else:
            self.end_iter = resume_epoch * self.iterations_per_epoch + max_epoch_or_iter
        self.current_iter = self.start_iter
        self.current_epoch = resume_epoch

        self.step_index = 0
        for ladder in self.stepvalues:
            if self.current_iter >= ladder:
                self.step_index += 1
            else:
                break

        self.batch_size_per_processor = int(batch_size / world_size)
        self.num_workers_per_processor = int((num_workers + world_size - 1) / world_size)

    def mayContinue(self):
        #print('self.current_iter=', self.current_iter)
        return self.current_iter < self.end_iter

    def update(self):
        self.current_iter +=1
        if self.current_iter in self.stepvalues:
            self.step_index += 1
        self.current_epoch = self.current_iter // self.iterations_per_epoch

    def needReloadData(self):
        return self.current_iter % self.iterations_per_epoch == 0

    def needSave(self, local_rank):
        #return False
        if (local_rank == 0):
            #return True # TODO: delete this line, it is only for debugging the DataLoader
            return (self.current_epoch % 10 == 0 and self.current_epoch > 0) or (self.current_epoch % 5 ==0 and self.current_epoch > 200)
        return False

    def needPrintLoass(self):
        return True
        #return self.current_iter % 10 == 0

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        if self.current_epoch < 6:
            lr = 1e-6 + (args.lr-1e-6) * self.current_iter / (self.iterations_per_epoch * 5)
        else:
            lr = args.lr * (self.gamma ** (self.step_index))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# ingredients: args.version, args.dataset_type, args.resume, args.resume_epoch
#              or args.basenet
def get_checkpoint_path():
    if args.resume:
        if args.resume_path == '':
            checkpoint_path = os.environ['PYTORCH_TRAIN_CHECKPOINT'] + "/rfbnet/checkpoints_fp/" + args.version + '_' + args.dataset + '_epoches_{:d}.pth'
        else:
            checkpoint_path = args.resume_path + "/" + args.version + '_' + args.dataset + '_epoches_{:d}.pth'
        checkpoint_path = checkpoint_path.format(args.resume_epoch)
    else:
        checkpoint_path = args.basenet
    return checkpoint_path

def load_checkpoint(resume):
    net_dict = None
    optimizer_dict = None
    mixed_dict = None

    relative_data_path = get_checkpoint_path()

    if not args.resume:
        net_dict = torch.load(relative_data_path)
    else:
        checkpoint = torch.load(relative_data_path, map_location='cpu')
        #state_dict = torch.load(relative_data_path)

        checkpoint_format = 0 # the original saving method
        if ("checkpoint_format" in checkpoint):
            checkpoint_format = checkpoint['checkpoint_format']

        if checkpoint_format == 1:
            net_dict = checkpoint['model']
            optimizer_dict = checkpoint['optimizer']
            if args.cnmix:
                mixed_dict = checkpoint['cnmix']
            elif args.pyamp:
                mixed_dict = checkpoint['amp']
        else: # checkpoint_format == 0
            net_dict = checkpoint

    return net_dict, optimizer_dict, mixed_dict

def load_net(net, net_dict, resume):

    if not resume:
        net.base.load_state_dict(net_dict)

        def xavier(param):
            init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0

        print('Initializing weights...')
    # initialize newly added layers' weights with kaiming_normal method
        net.extras.apply(weights_init)
        net.loc.apply(weights_init)
        net.conf.apply(weights_init)
        net.Norm.apply(weights_init)
        if args.version == 'RFB_E_vgg':
            net.reduce.apply(weights_init)
            net.up_reduce.apply(weights_init)

    else:
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in net_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        #net.eval()
        print('Finished loading model!')

    #print(net)
    return net

def save_net(net, optimizer, dataset_type, current_epoch, is_final_epoch):
    print('save_net')
    checkpoint = {"checkpoint_format":1,
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict()}
    if not is_final_epoch:
        print('path=', args.save_folder+'/'+args.version+'_'+ dataset_type + '_epoches_'+ repr(current_epoch) + '.pth')
        torch.save(checkpoint, args.save_folder+'/'+args.version+'_'+ dataset_type + '_epoches_'+
                            repr(current_epoch) + '.pth')
    else:
        print('path=', args.save_folder + '/' + 'Final_' + args.version +'_' + dataset_type+ '.pth')
        torch.save(checkpoint, args.save_folder + '/' +
               'Final_' + args.version +'_' + dataset_type+ '.pth')

def save_net_with_mixed(net, optimizer, dataset_type, current_epoch, is_final_epoch):
    print('save_net_with_mixed_precision')
    checkpoint = {"checkpoint_format":1,
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict()}
    if args.cnmix:
        checkpoint["cnmix"] = cnmix.state_dict()
    elif args.pyamp and scaler is not None:
        checkpoint["amp"] = scaler.state_dict()

    if not is_final_epoch:
        print('path=', args.save_folder+'/'+args.version+'_'+ dataset_type + '_epoches_'+
                            repr(current_epoch) + '.pth')
        torch.save(checkpoint, args.save_folder+'/'+args.version+'_'+ dataset_type + '_epoches_'+
                            repr(current_epoch) + '.pth')
    else:
        print('path=', args.save_folder + '/' +
               'Final_' + args.version +'_' + dataset_type+ '.pth')
        torch.save(checkpoint, args.save_folder + '/' +
               'Final_' + args.version +'_' + dataset_type+ '.pth')

def train(net, dataset_wrapper, training_strategy, optimizer, criterion, priors, distributed, local_rank, world_size):
    net.train()
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_wrapper.dataset)
    else:
        train_sampler = None

    #lr = args.lr
    metric_collector = MetricCollector(
        enable_only_benchmark=True,
        record_elapsed_time=True,
        record_hardware_time=True if args.device == 'mlu' else False)
    metric_collector.place()

    while training_strategy.mayContinue():
        if training_strategy.needReloadData():
            # create batch iterator
            if distributed:
                train_sampler.set_epoch(training_strategy.current_epoch)
            batch_iterator = iter(data.DataLoader(dataset_wrapper.dataset, training_strategy.batch_size_per_processor, sampler=train_sampler,
                                                  shuffle=(train_sampler is None), num_workers=training_strategy.num_workers_per_processor, collate_fn=detection_collate))

            if training_strategy.needSave(local_rank): # applies to both distributed or non-distributed
                if args.cnmix:
                    save_net_with_mixed(net, optimizer, dataset_wrapper.dataset_type, training_strategy.current_epoch, False)
                else:
                    save_net(net, optimizer, dataset_wrapper.dataset_type, training_strategy.current_epoch, False)
        load_t0 = time.time()

        lr = training_strategy.adjust_learning_rate(optimizer)

        # load train data
        images, targets = next(batch_iterator)
        if _USE_GPU:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]
        elif _USE_MLU:
            images = Variable(images.to(ct.mlu_device()))
            targets = [Variable(anno.to(ct.mlu_device())) for anno in targets]
        else: # cpu
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]
            
        # forward
        with autocast(enabled=args.pyamp):
            out = net(images)
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, priors, targets)
            loss = loss_l + loss_c
        if _USE_MLU and args.cnmix:
            with cnmix.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        elif args.pyamp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if args.pyamp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        load_t1 = time.time()

        # End 2 End time
        print('Epoch:' + repr(training_strategy.current_epoch) + ' || epochiter: ' + repr(training_strategy.current_iter % training_strategy.iterations_per_epoch) + '/' + repr(training_strategy.iterations_per_epoch)
                + '|| Total iter ' +
                repr(training_strategy.current_iter) + ' || L: %.4f C: %.4f||' % (
            loss_l.item(),loss_c.item()) +
            'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))

        # MetricCollector record
        metric_collector.record()
        metric_collector.place()
        
        training_strategy.update()
    # insert metrics and dump metrics
    if args.cnmix:
        precision = args.opt_level
    elif args.pyamp:
        precision = "amp"
    else:
        precision = "fp32"
    metric_collector.insert_metrics(
        net = "RFBNet",
        batch_size = int(args.batch_size/args.nprocessor),
        precision = precision,
        cards = args.nprocessor,
        DPF_mode = "ddp " if args.distributed == True else "single")

    if (args.distributed == False) or (local_rank == 0):
        metric_collector.dump()

    if local_rank == 0:  # applies to both distributed or non-distributed
        if args.cnmix or args.pyamp:
            save_net_with_mixed(net, optimizer, dataset_wrapper.dataset_type, training_strategy.current_epoch, True)
        else:
            save_net(net, optimizer, dataset_wrapper.dataset_type, training_strategy.current_epoch, True)

if __name__ == '__main__':
    #print('main(),',  os.getpid())
    if args.world_size>1 or args.nprocessor>1:
        mp.spawn(main_worker, nprocs=args.nprocessor, args=(args.nprocessor, args))
    else:
        main_worker(0, args.nprocessor, args) # for single process, the default rank is deemed as 0
