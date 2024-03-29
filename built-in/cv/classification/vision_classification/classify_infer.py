from __future__ import division
import os
import sys
import time
import logging
import argparse
import random
import warnings
import json
import re
from enum import Enum

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
from PIL import Image

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../tools/utils/")
from metric import MetricCollector
from common_utils import (
    get_precision_mode,
)

torch.set_grad_enabled(False)
env_dist = os.environ
#configure logging path
logging.basicConfig(level=logging.INFO,
                    format= '%(asctime)s - %(pathname)s[line:%(lineno)d] - ' \
                            '%(levelname)s: %(message)s')
logger = logging.getLogger("TestNets")

MAX_WORKSPACE_SIZE = 1 << 32

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_args():
    parser = argparse.ArgumentParser(description='Pre-checkin and Daily test script.')
    parser.add_argument("--batch_size", dest = "batch_size", help =
                        "batch size for one inference.",
                        default = 1,type = int)
    parser.add_argument("--input_data_type", dest = 'input_data_type', type = str, default="float32",
                        choices = ['float32', 'float16'])
    parser.add_argument("--network", dest = 'network', help =
                        "the network that will be running.",
                        default = "", type = str)
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading works (default: 4)')
    parser.add_argument('--iters', default = -1, type = int)
    parser.add_argument('--warmup_iters', default = 200, type = int)
    parser.add_argument('--device', default='cpu', type=str, choices = ['cpu', 'mlu', 'gpu'])
    parser.add_argument('--data', type=str, help = 'imagenet validation dir')
    parser.add_argument('-p', '--print-freq', default=1, type=int,
                        metavar='N', help='print frequency (default: 1)')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--qint', default='no_quant', dest='qint', choices = ['int8', 'int16', 'no_quant'])
    parser.add_argument('--quant_batch_num', default = 5, dest='quant_batch_num', type=int,
                        help='Set image numbers to evaluate quantized params, default is 5.')
    parser.add_argument('--ckpt', type=str, help = "model checkpoint file")
    parser.add_argument('--only_genoff', help = "only generate torchscript model", default = False, type=str2bool)
    parser.add_argument('--offline_model_path', help = "torchscript offline model path", type = str)
    parser.add_argument('--do_benchmark', help = "do benchmark test", default = False, type = str2bool)

    return parser.parse_args()

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, bs, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""""
    with torch.no_grad():
        output = output[:bs]
        target = target[:bs]
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = correct.contiguous()
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test_cls_network(args):
    net=None
    in_h, in_w, resize, crop = (224,224,256,224)
    net_name = args.network
    pretrained = True if args.ckpt is None else False
    if net_name == "shufflenet_v2_x1_5":
        pretrained = False
    if net_name == 'inception_v3':
        net = getattr(models, net_name)(pretrained=pretrained, transform_input=False)
        in_h, in_w, resize, crop = (299,299,299,299)
    elif net_name == 'googlenet':
        net = getattr(models, net_name)(pretrained=pretrained, transform_input=False, aux_logits = False)
        # set googlenet aux as sigmoid op is for the success of the torch.jit.trace call
        net.aux1 = torch.nn.Sigmoid()
        net.aux2 = torch.nn.Sigmoid()
        in_h, in_w, resize, crop = (299,299,299,299)
    elif net_name == 'alexnet':
        net = getattr(models, net_name)(pretrained=pretrained)
        in_h, in_w, resize, crop = (227,227,256,227)
    else:
        net = getattr(models, net_name)(pretrained=pretrained)
    if args.ckpt is not None:
        pretrained_ckpt = torch.load(args.ckpt)
        if 'state_dict' in pretrained_ckpt.keys():
            pretrained_ckpt = pretrained_ckpt['state_dict']
        if net_name in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$'
            )
            for key in list(pretrained_ckpt.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    pretrained_ckpt[new_key] = pretrained_ckpt[key]
                    del pretrained_ckpt[key]
            net.load_state_dict(pretrained_ckpt)
        elif net_name == "googlenet":
            net = models.GoogLeNet(transform_input=False, aux_logits = True)
            net.load_state_dict(pretrained_ckpt)
            net.aux_logits = False
            net.aux1 = torch.nn.Sigmoid()
            net.aux2 = torch.nn.Sigmoid()
        else:
            net.load_state_dict(pretrained_ckpt)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if net_name == 'googlenet':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    net.eval()

    print("begin loading dataset...")
    valdir = os.path.join(args.data, 'val')
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
                            transforms.Resize(resize),
                            transforms.CenterCrop(crop),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std)
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers
    )
    calib_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
                            transforms.Resize(resize),
                            transforms.CenterCrop(crop),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std)
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers
    )
    if args.device == 'mlu':
        device = torch.device('mlu')
    elif args.device == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = net.to(device)
    if args.input_data_type == "float16":
        model = model.half()

    batch_time = AverageMeter('Batch Time', ':6.3f', Summary.AVERAGE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    if args.do_benchmark:
        metric_collector = MetricCollector(
            record_elapsed_time=True,
            record_hardware_time=True if args.device == 'mlu' else False
        )
        input_data = torch.randn(args.batch_size, 3, in_h, in_w).to(device)
        if args.input_data_type == "float16":
            input_data = input_data.half()
        assert args.iters > 0, "iterations of benchmark test must greater than 0"
        assert args.iters > args.warmup_iters, "iters must greater than warmup iters"
        for i in range(args.warmup_iters):
            model(input_data)
        # do synchronize()
        if args.device == 'mlu':
            torch.mlu.synchronize()
        elif args.device == 'cuda':
            torch.cuda.synchronize()

        progress_perf = ProgressMeter( len(val_loader), [batch_time], prefix='Test: ')
        end = time.time()
        for i in range(args.iters - args.warmup_iters):
            metric_collector.place()
            model(input_data)
            if args.device == 'mlu':
                torch.mlu.synchronize()
            elif args.device == 'cuda':
                torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()
            metric_collector.record()
            if i % args.print_freq == 0:
                progress_perf.display(i)
        metric_collector.insert_metrics(
            net = args.network,
            batch_size = args.batch_size,
            precision = get_precision_mode(args.input_data_type, args.qint),
            cards = 1,
            DPF_mode = "single"
        )
        progress_perf.display_summary()
        metric_collector.dump()
        sys.exit(0)
    progress_accuracy = ProgressMeter( len(val_loader), [top1, top5], prefix='Test: ')
    for i, (images, target) in enumerate(val_loader):
        if i == args.iters:
            break
        bs = images.size(0)
        if images.size(0) < args.batch_size:
            for index in range(images.size(0) + 1, args.batch_size + 1):
                images = torch.cat((images, images[0].unsqueeze(0)), 0)
                target = torch.cat((target, target[0].unsqueeze(0)), 0)

        images = images.to(device, dtype = torch.float16 if args.input_data_type == "float16" else torch.float32)
        target = target.to(device)
        output = model(images)
        acc1, acc5 = accuracy(output, target, bs, topk=(1, 5))
        top1.update(acc1[0], bs)
        top5.update(acc5[0], bs)
        # measure elapsed time


        if i % args.print_freq == 0:
            progress_accuracy.display(i)

    progress_accuracy.display_summary()
    metric_collector = MetricCollector()
    metric_collector.insert_metrics(
        net = args.network,
        batch_size = args.batch_size,
        precision = get_precision_mode(args.input_data_type, args.qint),
        accuracy = [top1.avg.item(), top5.avg.item()]
    )
    metric_collector.dump()

if __name__ == '__main__':
    args = get_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # set pretrained model path
    TORCH_HOME = os.getenv('TORCH_HOME')
    if TORCH_HOME == None:
        print("Warning: please set environment variable TORCH_HOME")
        exit(1)
    torch.hub.set_dir(os.getenv('TORCH_HOME'))

    assert args.network != "", "please set network"
    if args.device == 'mlu':
        import torch_mlu

    test_cls_network(args)

