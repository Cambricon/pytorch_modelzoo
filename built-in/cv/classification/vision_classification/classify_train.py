import argparse
import copy
import os
import re
import random
import shutil
import sys
import time
import warnings
import math
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import pandas as pd
from collections import OrderedDict
from torch.cuda.amp import autocast, GradScaler

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../tools/utils/")
from metric import MetricCollector

import argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-p', '--print-freq', default=1, type=int,
                        metavar='N', help='print frequency (default: 1)')
parser.add_argument('-m', '--modeldir', type=str, default='./', metavar='DIR',
                        help='path to dir of models and mlu operators, default is ./ and from torchvision')
parser.add_argument('--data', default="./imagenet",
                        type=str, metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading works (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
parser.add_argument('--resume_multi_device', action='store_true',
                        help='Only when model is saved by gpu distributed, enable this to load model with submodule')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
parser.add_argument("--save_ckp", dest='save_ckp', action='store_true',
                        help="Enable save checkpoint")
parser.add_argument('--iters', type=int, default=30000, metavar='N',
                        help='iters per epoch')
parser.add_argument('--device', default='cpu', type=str,
                        help='Use cpu gpu or mlu device')
parser.add_argument('--device_id', default=None, type=int,
                        help='Use specified device for training, useless in multiprocessing distributed training')
parser.add_argument('--pretrained', dest="pretrained", action="store_true",
                        help="Use a pretrained model")
parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
parser.add_argument('--ckpdir',type=str,default='./ckps',metavar='DIR',
                        help='Where to save ckps')
parser.add_argument('--logdir',type=str,default='./log_mlu',metavar='DIR',
                        help='Where to save logs')
parser.add_argument('--hvd', type=int, default=-1,
                        help='how manys cards if using horovod')
parser.add_argument('--cnmix', action='store_true', default=False,
                    help='use cnmix for mixed precision training')
parser.add_argument('--opt_level', type=str, default='O1',
                        help='choose level of mixing precision')
parser.add_argument('--dummy_test', dest='dummy_test', action='store_true',
                        help='use fake data to traing')
parser.add_argument('--pyamp', action='store_true', default=False,
                    help='use pytorch amp for mixed precision training')
parser.add_argument('--start_eval_at', dest='start_eval_at', type=int, default=None, 
                    help='start evaluation at specified epoch')
parser.add_argument('--evaluate_every', '--eval_every', dest='evaluate_every', type=int, default=None,
                    help='evaluate at every epochs')
parser.add_argument('--quality_threshold', dest='quality_threshold', type=float, default=None, 
                    help='target accuracy')


model_path = parser.parse_known_args()[0].modeldir
sys.path.append(model_path)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

args = parser.parse_args()
if args.device == 'mlu':
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
elif args.hvd != -1 or args.cnmix:
    print("MLU hvd and cnmix can not be used without MLU currently!!!!")
    sys.exit(1)

if args.hvd != -1:
    import horovod.torch as hvd
    hvd.init()

if args.cnmix:
    import cnmix

class dummy_data_loader():
    def __init__(self, len = 0, images_size = (3, 224, 224), batch_size = 1, num_classes = 1000):
        self.len = len
        images = torch.normal(mean = -0.03 , std = 1.24, size = (batch_size,)+images_size)
        target = torch.randint(low = 0, high = num_classes, size = (batch_size,))
        self.images = images.to(ct.mlu_device(), non_blocking=True)
        self.target = target.to(ct.mlu_device(), non_blocking=True)
        self.data = 0
    def __iter__(self):
        return self
    def __len__(self):
        return self.len
    def __next__(self):
        if self.data > self.len:
            raise StopIteration
        else:
            self.data += 1
            return self.images, self.target

def main():
    args.start_epoch=0

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.multiprocessing_distributed  or args.world_size > 1

    if args.hvd != -1:
        args.device_id = hvd.local_rank()

    ndevs_per_node = ct.device_count() if args.device == 'mlu' else torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ndevs_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ndevs_per_node, args=(ndevs_per_node, args))
    else:
        main_worker(args.device_id, ndevs_per_node, args)

def main_worker(dev_id, ndevs_per_node, args):
    args.device_id = dev_id
    if args.device_id is None:
        args.device_id = 0  # Default Device is 0
    if args.device == 'mlu':
        ct.set_device(args.device_id)
        if args.hvd != -1:
            args.rank = hvd.rank()
        print("Use MLU{} for training".format(args.device_id))
    elif args.device == 'gpu':
        torch.cuda.set_device(args.device_id)
        print("Use GPU{} for training".format(args.device_id))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ndevs_per_node + dev_id
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                rank=dev_id, world_size=ndevs_per_node)

    acc_all   = []
    time_all  = []
    loss_all  = []
    epoch_all = []

    acc_all_val   = []
    time_all_val  = []
    loss_all_val  = []
    epoch_all_val = []

    # Data Loader:
    print ("=> loading dataset")
    traindir      = os.path.join(args.data, 'train')
    valdir        = os.path.join(args.data, 'val')
    normalize     = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(traindir,
                             transforms.Compose([transforms.RandomResizedCrop(224),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 normalize,]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        if args.hvd != -1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        else:
            train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
                ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers) #, pin_memory=True)

    #Create Model:
    if args.pretrained:
        print("=> Using pre-trained model: {}".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else :
        print("=> Creating Model: {}".format(args.arch))
        model = models.__dict__[args.arch]()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scaler = None
    if args.pyamp:
        scaler = GradScaler()
    #Resume from Checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint: {}".format(args.resume))
            resume_point = torch.load(args.resume, map_location=torch.device('cpu'))
            #print(resume_point['state_dict'].keys())
            resume_point_replace = {}
            if args.resume_multi_device: # DDP module create by multi device
                # Remove "submodule" (e.g model.submodule.conv1 -> model.conv1)
                # and "module" (e.g features.module.conv2d -> features.conv2d)
                # they are created during DDP training, different from origin model
                for key in resume_point['state_dict'].keys():
                    split_key = key.split('.')
                    split_origin = copy.deepcopy(split_key)
                    for item in split_origin:
                        if item == "module":
                            split_key.remove("module")
                        elif item == "submodule":
                            split_key.remove("submodule")
                    resume_point_replace[".".join(split_key)] = resume_point['state_dict'][key]
            else:
                resume_point_replace = resume_point['state_dict']
            args.start_epoch = resume_point['epoch']
            print("Resume from epoch {}".format(args.start_epoch))
            model.load_state_dict(resume_point_replace, strict=True if args.device=='gpu' else False)
            resume_optimizer = resume_point['optimizer']
            if args.pyamp:
                if isinstance(resume_point, dict) and 'amp' in resume_point:
                    scaler.load_state_dict(resume_point['amp'])
        else:
            print("ERROR: Fail to load Resume checkpoint from {}, file not exist".format(args.resume))
            return

    if args.device == 'mlu':
        model.to(ct.mlu_device())
    elif args.device == 'gpu':
        model.to(torch.device("cuda"))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume: # Resume optimizer
        optimizer.load_state_dict(resume_optimizer)

    if args.hvd != -1:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    if args.device == 'mlu' and args.cnmix:
        if args.arch in ["shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5"]:
            cnmix.core.cnmix_set_amp_use_online(True)
        model, optimizer = cnmix.initialize(model, optimizer, opt_level=args.opt_level)
        if args.resume:
            if os.path.isfile(args.resume):
               checkpoint = torch.load(args.resume, map_location='cpu')
               if isinstance(checkpoint, dict) and 'cnmix' in checkpoint:
                   cnmix.load_state_dict(checkpoint['cnmix'])
    if args.distributed:
        model = DDP(model, device_ids=[args.device_id])

    model.train()

    if args.device == 'mlu':
        criterion.to(ct.mlu_device())
        ct.to(optimizer, torch.device('mlu'))
    elif args.device == 'gpu':
        criterion.to(torch.device("cuda"))

    if args.evaluate:
        print("=> Test on val-dataset only")
        validate(val_loader, model, criterion, args)
        return

    if args.device == 'mlu' and args.cnmix:
       cnmix.cnmix_set_amp_quantify_params('all', {'batch_size': args.batch_size,
                                                   'data_num': args.batch_size * len(train_loader)})

    next_eval_at = args.start_eval_at
    # Train epochs, We save epoch at the start, to make sure DDP-Reduce finished on each Process
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.save_ckp == 1:
            if args.device == 'mlu':
                if (args.distributed == False and args.hvd == -1) or (args.rank == 0): # Only save checkpoint by Process 0
                    if not os.path.exists(args.ckpdir):
                        os.makedirs(args.ckpdir)
                    save_file_path = os.path.join(args.ckpdir, args.arch + "_" + str(epoch) + ".pth")
                    print("=> Save file to {}".format(save_file_path))
                    if args.distributed:
                        checkpoint = {"state_dict":model.module.state_dict(), "optimizer":optimizer.state_dict(),
                                      "epoch": epoch}
                    else:
                        checkpoint = {"state_dict":model.state_dict(), "optimizer":optimizer.state_dict(),
                                      "epoch": epoch}
                    if args.cnmix:
                        checkpoint["cnmix"]=cnmix.state_dict()
                    if args.pyamp and scaler is not None:
                        checkpoint["amp"]=scaler.state_dict()
                    torch.save(checkpoint, save_file_path)
                    print("=> Model save finished")
                    # Load from ckp:
            elif args.device == 'gpu':
                if args.distributed == False or args.rank == 0:
                    if not os.path.exists(args.ckpdir):
                        os.makedirs(args.ckpdir)
                    save_file_path = os.path.join(args.ckpdir, args.arch + "_" + str(epoch) + ".pth")
                    print("=> Save file to {}".format(save_file_path))
                    if args.distributed:
                        checkpoint = {"state_dict":model.module.state_dict(), "optimizer":optimizer.state_dict(),
                                      "epoch": epoch}
                    else:
                        checkpoint = {"state_dict":model.state_dict(), "optimizer":optimizer.state_dict(),
                                      "epoch": epoch}
                    torch.save(checkpoint, save_file_path)
                    print("=> Model save finished")

        # Train: Skip last epoch
        if epoch < args.epochs:
            if args.arch not in ["mobilenet_v2", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5"]:
                adjust_learning_rate(optimizer, epoch, args)
            if args.distributed or args.hvd != -1:
                train_sampler.set_epoch(epoch)
            train_metrics = train(train_loader, model, criterion, optimizer, epoch, scaler, args)
            if args.quality_threshold is not None:
                if epoch == next_eval_at:
                    print("=> Test on val-dataset only")
                    next_eval_at += args.evaluate_every
                    result = validate(val_loader, model, criterion, args)
                    print('top1 {}'.format(result['top1']))
                    if result['top1'] >= args.quality_threshold:
                        print("top1 {} achieved, training finished".format(result['top1']))
                        break

def train(train_loader, model, criterion, optimizer, epoch, scaler, args):
    adaptive_cnt = int(os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT')) if (os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT') is not None) else 0
    batch_time_benchmark = []
    batch_time = AverageMeter('Time' , ':6.3f')
    data_time  = AverageMeter('Data' , ':6.3f')
    losses     = AverageMeter('Loss' , ':.4e' )
    top1       = AverageMeter('Acc@1', ':6.2f')
    
    pid_num = os.getpid()

    progress   = ProgressMeter(
                   len(train_loader),
                   [ batch_time, data_time, losses, top1, pid_num ],
                   prefix='[{}]'.format(epoch))

    loss_columns = []
    acc_columns  = []
    time_columns = []
    iter_columns = []

    # switch to train mode
    model.train()
    end = time.time()
    if args.dummy_test:
        train_loader = dummy_data_loader(len = len(train_loader), batch_size = args.batch_size)
    # for internal benchmark test
    metric_collector = MetricCollector(
        enable_only_benchmark=True,
        record_elapsed_time=True,
        record_hardware_time=True if args.device == 'mlu' else False)
    metric_collector.place()

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() -end)
        if args.arch == "mobilenet_v2":
            adjust_learning_rate_cos(optimizer, epoch, i, len(train_loader), args)
        if args.arch in ["shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5"]:
            adjust_learning_rate_poly_warmup(optimizer, epoch, i, len(train_loader), args)
        if i == args.iters:
            break
        if not args.dummy_test:
            images = Variable(images.float(), requires_grad=False)
            if args.device == 'gpu':
                images = images.cuda(args.device_id, non_blocking=True)
                target = target.cuda(args.device_id, non_blocking=True)
            elif args.device == 'mlu':
                images = images.to(ct.mlu_device(), non_blocking=True)
                target = target.to(ct.mlu_device(), non_blocking=True)
        if args.arch == 'googlenet':
            with autocast(enabled=args.pyamp):
                aux1, aux2, output = model(images)
                loss1 = criterion(output, target)
                loss2 = criterion(aux1, target)
                loss3 = criterion(aux2, target)
                loss = loss1 + 0.3 * (loss2 + loss3)
        else:
            with autocast(enabled=args.pyamp):
                output = model(images)
                loss = criterion(output, target)
        #measure accuracy and loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        #compute gradient and do SGD step
        optimizer.zero_grad()
        if args.device == 'mlu' and args.cnmix:
           with cnmix.scale_loss(loss, optimizer) as scaled_loss:
               scaled_loss.backward()
               if args.hvd != -1:
                   optimizer.synchronize()
        elif args.pyamp:
            scaler.scale(loss).backward()
        else:
           loss.backward()

        if args.hvd != -1 and args.cnmix:
            with optimizer.skip_synchronize():
                optimizer.step()
        elif args.pyamp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        loss_item = loss.item()
        # MetricCollector record
        metric_collector.record()
        metric_collector.place()
        # End 2 End time
        if i >= adaptive_cnt:
            batch_time_benchmark.append(time.time() - end)
        batch_time.update(time.time() - end)
        end = time.time()
        loss_columns.append(loss_item)
        acc_columns.append(acc1[0].cpu().numpy())
        time_columns.append(time.time() - end)
        iter_columns.append(int(i))

        #LOG
        if i % args.print_freq == 0:
            progress.display(i)
        if not os.path.exists(os.path.join(args.logdir)):
            try:
                os.makedirs(os.path.join(args.logdir))
            except:
                print("INFO: Multiprocesses make dirs")
        train_f = open(args.logdir + '/epoch_' + str(epoch) + '_' + str(args.iters) + '_' + str(args.rank) + '.csv', 'a')
        train_f.write('{},{},{},{}\n'.format(iter_columns[-1], loss_columns[-1], acc_columns[-1], time_columns[-1]))
        train_f.close()
    # insert metrics and dump metrics
    if args.pyamp:
        precision = "amp"
    elif args.cnmix:
        precision = args.opt_level
    else:
        precision = "fp32"
    metric_collector.insert_metrics(
        net = args.arch,
        batch_size = args.batch_size,
        precision = precision,
        cards = ct.device_count() if args.rank == 0 else 1,
        DPF_mode = "ddp " if args.multiprocessing_distributed == True else "single")
    if ((args.distributed == False and args.hvd == -1) or (args.rank == 0)):
        metric_collector.dump()

    return OrderedDict([('loss', loss.item()), ('top1', top1.avg)])


def validate(val_loader, model, criterion, args, epoch=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    loss_columns=[]
    acc_columns=[]
    time_columns=[]
    iter_columns=[]

    model.eval()
    with torch.no_grad():
        end = time.time()
        total = time.time()
        for i, (images, target) in enumerate(val_loader):
            if i == args.iters:
                break
            if args.device == 'gpu':
                images = images.cuda(args.device_id, non_blocking=True)
                target = target.cuda(args.device_id, non_blocking=True)
            if args.device == 'mlu':
                images = images.to("mlu:{}".format(args.device_id), non_blocking=True)
                target = target.to("mlu:{}".format(args.device_id), non_blocking=True)

            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

        # this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        metric_collector = MetricCollector(enable_only_avglog=True)
        metric_collector.insert_metrics(net = args.arch,
                                        accuracy = [top1.avg.item(), top5.avg.item()])
        metric_collector.dump()

        loss_columns.append(loss.item())
        acc_columns.append(acc1[0].cpu().numpy())
        time_columns.append(time.time()-total)
        iter_columns.append(int(i))

    csv_save=pd.DataFrame(columns=['iter','loss','acc','time'],data=np.transpose([iter_columns,loss_columns,acc_columns,time_columns]))
    loss_location=(args.logdir)
    csv_save.to_csv(loss_location + '/epoch_'+str(epoch)+'_val.csv')

    return OrderedDict([('loss', loss.item()), ('top1', top1.avg), ('top5', top5.avg)])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.arch == "alexnet":
        lr = float(args.lr) * (0.94 ** (epoch // 2))
    else:
        lr = float(args.lr) * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_cos(optimizer,epoch,iteration,num_iter,args):
    lr = optimizer.param_groups[0]['lr']

    #warmup_epoch =5 if args.warmup else 0
    warmup_epoch = 3
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration+epoch *num_iter
    max_iter = args.epochs * num_iter

    lr =args.lr * (1 + math.cos(math.pi*(current_iter-warmup_iter)/(max_iter-warmup_iter)))/2

    if epoch < warmup_epoch:
        lr=args.lr*current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_poly_warmup(optimizer, epoch, now_iter, num_iter, args):
    #warmup: during warmup_epochs, lr is args.lr * args.warmup_ratio
    #lr_decay: lr = (args.lr - min_lr) * coeff + min_lr
    #coeff: coeff = (1 - (iter - warmup_iter) / max_iter) ** args.power
    # TODO Given params ONLY for shufflenet
    current_iter = now_iter+epoch *num_iter
    warmup_epochs=4
    power = 1
    min_lr = 0
    warmup_ratio=0.1
    if epoch < warmup_epochs:
        lr = args.lr * warmup_ratio
    else:
        max_iter = args.epochs * num_iter
        warmup_iter = warmup_epochs * num_iter
        coeff = (1 - (current_iter - warmup_iter) / max_iter) ** power
        lr = (args.lr - min_lr) * coeff + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    start_time=time.time()
    main()
    use_time=time.time()-start_time
    print('use time' , use_time)

