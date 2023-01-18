from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
from utils import *
from dataset import *
import copy
import sys
import crnn as crnnmodel

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../tools/utils/")
from metric import MetricCollector

try:
    if torch.is_mlu_available():
        import torch_mlu
        import torch_mlu.core.mlu_model as ct
except:
    print("Cambricon CATCH is not available!")

parser = argparse.ArgumentParser()
parser.add_argument('--valRoot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--mlu', action='store_true', help='enables mlu')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--expr_dir', default='./checkpoint_8', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--cudnn_lstm', action='store_true', help='if GPU, using cudnn LSTM; if MLU, using cnnl LSTM; otherwise using multi-operators')
parser.add_argument('--infr_iter', type=int, default=1000, help='number of iters to infr for')

def main():
    opt = parser.parse_args()
    print(opt)
    if not os.path.exists(opt.expr_dir):
        os.makedirs(opt.expr_dir)

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    try:
        if torch.is_mlu_available() and not opt.mlu:
            print("WARNING: You have a MLU device, so you should probably run with --mlu")
    except:
        print("Cambricon CATCH is not available!")
    # gpu=1
    # device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(1)

    batch_size = opt.batchSize
    test_dataset = lmdbDataset(
        root=opt.valRoot, transform=resizeNormalize((100, 32)))

    nclass = len(opt.alphabet) + 1
    nc = 1

    opt.converter = strLabelConverter(opt.alphabet)
    try:
        from warpctc_pytorch import CTCLoss
        criterion = CTCLoss()
    except:
        print("WARNING: warpctc is not supported !")
        from torch.nn import CTCLoss
        cudnn.enabled = False
        criterion = torch.nn.CTCLoss(reduction='sum', zero_infinity=True)

    if opt.cudnn_lstm:
        print("Using cudnn/cnnl LSTM!")
        cudnn.enabled = True
        cudnn.set_flags(True, False, False)
    else:
        cudnn.enabled = True
        cudnn.set_flags(False, False, False)

    crnn = crnnmodel.CRNN(opt.imgH, nc, nclass, opt.nh)
    model_path = opt.pretrained
    pretrained_weights = torch.load(model_path, map_location=torch.device('cpu'))
    pretrained_weights_replace = {}
    if 'state_dict' in pretrained_weights:
        for key in pretrained_weights['state_dict'].keys():
            split_key = key.split('.')
            split_origin = copy.deepcopy(split_key)
            for item in split_origin:
                if item == "module":
                    split_key.remove("module")
                elif item == "submodule":
                    split_key.remove("submodule")
            pretrained_weights_replace[".".join(split_key)] = pretrained_weights['state_dict'][key]
    else:
        pretrained_weights_replace = {
            k.replace("module.", ""): v for k, v in pretrained_weights.items()
        }

    if opt.cuda:
        crnn = crnn.to(0, non_blocking=True)
    elif opt.mlu:
        crnn = crnn.to(ct.mlu_device(), non_blocking=True)
    print('loading pretrained model from %s' % model_path)
    # model.load_state_dict(torch.load(model_path))
    crnn.load_state_dict(pretrained_weights_replace)
    print(crnn)

    image = torch.FloatTensor(batch_size, 3, opt.imgH, opt.imgH)
    text = torch.IntTensor(batch_size * 5)
    length = torch.IntTensor(batch_size)

    if opt.cuda:
        crnn.to(0, non_blocking=True)
        image = image.to(0, non_blocking=True)
        criterion = criterion.to(0, non_blocking=True)
        crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    elif opt.mlu:
        crnn.to(ct.mlu_device(), non_blocking=True)
        image = image.to(ct.mlu_device(), non_blocking=True)
        criterion = criterion.to(ct.mlu_device(), non_blocking=True)

    opt.image = Variable(image)
    opt.text = Variable(text)
    opt.length = Variable(length)

    val(crnn, test_dataset, criterion, opt)


def val(net, dataset, criterion, opt):
    print('Start val')
    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers), pin_memory=True)
    val_iter = iter(data_loader)
    epoch_size = len(data_loader)

    n_correct = 0
    loss_avg = averager()
    print("epoch_size:",epoch_size)

    max_iter = min(opt.infr_iter, epoch_size)
    for i in range(max_iter):
        data = val_iter.next()
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        loadData(opt.image, cpu_images)
        t, l = opt.converter.encode(cpu_texts)
        loadData(opt.text, t)
        loadData(opt.length, l)

        if opt.mlu:
            preds = net(opt.image.to(ct.mlu_device(), non_blocking=True))
        else:
            preds = net(opt.image)
        preds_size = Variable(torch.IntTensor([preds.size(1)] * batch_size))
        preds = preds.permute(1, 0, 2)  # to use CTCloss format ?
        if opt.mlu:
            cost = torch.ops.torch_mlu.warp_ctc_loss(preds, opt.text.to(ct.mlu_device(), non_blocking=True), preds_size.to(ct.mlu_device(), non_blocking=True), opt.length.to(ct.mlu_device(), non_blocking=True), 0, 1, True, 0) / batch_size
        else:
            cost = criterion(preds, opt.text, preds_size, opt.length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        # preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = opt.converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = opt.converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))


    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuracy: %f' % (loss_avg.val(), accuracy))

    metric_collector = MetricCollector(enable_only_avglog=True)
    metric_collector.insert_metrics(net = "crnn", accuracy = accuracy)
    metric_collector.dump()


if __name__ == '__main__':
    main()

