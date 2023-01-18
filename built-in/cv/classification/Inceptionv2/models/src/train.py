import argparse
import math
import os
import random
import shutil
import time
import re

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

from tensorboardX import SummaryWriter

from utils import *

import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../../tools/utils/")
from metric import MetricCollector

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# device parametes
parser.add_argument('--use-cpu', default=None, action='store_true',
                    help='use cpu to train model.')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')


# train data parameters
parser.add_argument('--train-dataset', default=None,
                    help='path to train dataset')
parser.add_argument('--train-workers', default=None, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')


# valid data parameters
parser.add_argument('--valid-dataset', default=None,
                    help='path to valid dataset')
parser.add_argument('--valid-workers', default=None, type=int,
                    help='number of data loading workers (default: 4)')


# model parameters
parser.add_argument('--arch', metavar='ARCH', default=None,
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--config', default='config/alexnet_train.yaml',
                    help='model train and valid config')
parser.add_argument('--pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--pretrained-ckp', default=None,
                    help='pretrained checkpoint path')

# train parameters
parser.add_argument('--epochs', default=None, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch-size', default=None, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--learning-rate', default=None, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=None, type=float,
                    help='momentum')
parser.add_argument('--weight-decay', default=None, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', metavar='OPTIM', default=None,
                    choices=optimizer_names,
                    help='model architecture: ' +
                        ' | '.join(optimizer_names) +
                        ' (default: sgd)')
parser.add_argument('--scheduler', metavar='SCHEDULER', default=None,
                    choices=scheduler_names,
                    help='model architecture: ' +
                        ' | '.join(scheduler_names) +
                        ' (default: step)')
parser.add_argument('--warmup-epoch', default=None, type=int)
parser.add_argument('--scheduler-step', default=None, type=int,
                    help='scheduler step to adjust learning rate')
parser.add_argument('--scheduler-gamma', default=None, type=float,
                    help='scheduler scale.')
parser.add_argument('--criterion', metavar='CRITERION', default=None,
                    choices=criterion_names,
                    help='model architecture: ' +
                        ' | '.join(criterion_names) +
                        ' (default: cross_entropy)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

# distributed parameters
parser.add_argument('--world-size', default=None, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default=None, type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default=None, type=str,
                    help='distributed backend')
parser.add_argument('--distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# valid parameters
parser.add_argument('--valid-batch-size', default=None, type=int,
                    help='batch size of valid set.')

# log parameters
parser.add_argument('--ckpt-path', type=str, default='ckps',
                    help='path to save ckps')

# checkin parameters
parser.add_argument('--train_iterations', type=int, default=-1,
                    help='how many iterations to train')
parser.add_argument('--eval_iterations', type=int, default=-1,
                    help='how many iterations to validation')

# use mlu parameters
parser.add_argument('--device', type=str, default='cuda',
                    help='use device.')

# train with cnmix
parser.add_argument('--cnmix', action='store_true',
                    help='use cnmix.')
parser.add_argument('--opt_level', type=str,default='O1',
                    help='the level of cnmix.')
parser.add_argument('--dummy_test', dest='dummy_test', action='store_true',
                        help='use fake data to traing')
parser.add_argument('--pyamp', action='store_true', default=False,
                    help='use pytorch amp for mixed precision training')

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
# load arguments
args = parser.parse_args()
if args.device == 'mlu':
    import torch_mlu
    import torch_mlu.core.mlu_model as ct

def main():
    # load configs
    config = precess_train_config(args)
    # init seed
    init_seed(args.seed, not args.use_cpu)
    # prepare
    if args.use_cpu: # not support in cpu mode
        args.distribued = False
    else:
        if args.device == 'mlu':
            ngpus_per_node = ct.device_count()
        elif args.device == 'cuda':
            ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        config['train']['world_size'] = ngpus_per_node * config['train']['world_size']
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config))
    else:
        main_worker(args.gpu, ngpus_per_node, args, config)


def main_worker(gpu, ngpus_per_node, args, config):
    # prepare device
    args.gpu = gpu
    args.rank = gpu
    device = set_device(args.gpu, args.use_cpu, args.device)
    # init envs
    prefix = 'iters' if args.train_iterations>0 else 'logs'
    prefix = "{}_{}".format(prefix, config['model'])
    writer = set_writer(prefix, args.gpu, ngpus_per_node)
    if args.distributed:
        init_dist(config['train'], args.gpu, ngpus_per_node, device=args.device)

    # create model by default is not pretrained model
    model = create_model(
        config['model'], args.pretrained, config['pretrain']['path'], args).to(device)

    # Data load
    train_loader, train_sampler = create_loader(config['train_dataset'], config['train'], args.distributed,
                                                ndevs_per_node=ngpus_per_node, dev=gpu, device=args.device)
    val_loader, _ = create_loader(config['valid_dataset'], config['valid'], device=args.device)

    # create criterion, optimizer, scheduler
    criterion = create_criterion(config['train']['criterion']).to(device)
    optimizer = create_optimizer(model, config['train']['optimizer'])
    scheduler = Scheduler(config['train']['scheduler'], config['train']['optimizer']['learning_rate'], len(train_loader))
    batch_size = config['train']['batch_size']

    #use adaptive quantified strategy
    if args.device == 'mlu':
        model = model.to('mlu')

    if args.device == 'mlu' and  args.cnmix:
        import cnmix
        model, optimizer = cnmix.initialize(model, optimizer, opt_level=args.opt_level)
        cnmix.core.cnmix_set_amp_quantify_params('all', {'batch_size': batch_size,
                                                        'data_num': batch_size * len(train_loader)})

    # optionally resume from a checkpoint
    if args.resume:
        args.start_epoch, best_acc1 = resume_train(args.resume, model, optimizer, device, args.cnmix)
    else:
        best_acc1 = 0.0
    if args.distributed:
        model = convert_ddp_model(model, config['train']['rank'], device=args.device)

    # daily checkin
    if args.train_iterations > 0:
        iters = args.train_iterations
        args.ckpt_path = prefix
    else:
        iters = None

    scaler = None
    if args.pyamp:
        scaler = GradScaler()
        args.scaler = scaler

    for epoch in range(args.start_epoch, config['train']['scheduler']['epochs']):
        # random sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, epoch,  model, criterion, optimizer, scheduler, writer, device, batch_size, iters, args)

        # evaluate on validation set
        if os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT'):
            break
        acc1_val = validate(val_loader, epoch, model, writer, device, args.eval_iterations)

        # remember best acc@1 and save checkpoint
        is_best = acc1_val > best_acc1
        best_acc1 = max(acc1_val, best_acc1)

        # save checkpoint
        if not args.distributed or (args.distributed and config['train']['rank'] % ngpus_per_node ==0):
            state = {
                'epoch': epoch + 1,
                'model': model.module.state_dict() if type(model) == torch.nn.parallel.DistributedDataParallel else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc1': best_acc1,
            }
            if args.cnmix:
                state["cnmix"] = cnmix.state_dict()
            if args.pyamp and scaler is not None:
                state["amp"] = scaler.state_dict()
            if ((epoch+1) % (config['train']['scheduler']['epochs'] // 10)) == 0:
                save_ckpt(state, False, path='{}_{}'.format(args.ckpt_path, config['model']), epoch=epoch)
            save_ckpt(state, is_best, path='{}_{}'.format(args.ckpt_path, config['model']))
        if args.train_iterations > 0:
            break

    writer.close()


def train(train_loader, epoch, model, criterion, optimizer, scheduler, writer, device, batch_size, iters=None, args=None):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1],
        prefix="Train: [{}]".format(epoch))

    batch_time_m = AverageMeter('BatchTimeAve')
    data_time_m = AverageMeter('DataTimeAve')

    # switch to train mode
    model.train()

    end = time.time()
    train_iters = len(train_loader) * epoch
    if args.dummy_test:
        train_loader = dummy_data_loader(len = len(train_loader), batch_size = batch_size)

    # for internal benchmark test
    metric_collector = MetricCollector(
            enable_only_benchmark=True,
            record_elapsed_time=True,
            record_hardware_time=True if args.device == "mlu" else False)
    metric_collector.place()

    for i, (images, target) in enumerate(train_loader):

        # for daily checkin
        if iters and i >= iters:
            break

        data_time_m.update(time.time() - end)
        # adjust leaning rate
        adjust_lr(optimizer, scheduler.get_lr(epoch, train_iters+i))
        writer.add_scalar('train/lr', scheduler.cur_lr, train_iters+i)

        # set data to device
        if not args.dummy_test:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

        # compute output
        with autocast(enabled=args.pyamp):
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = calcu_class_acc(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.pyamp:
            scaler = args.scaler
        if args.device == 'mlu' and args.cnmix:
            import cnmix
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

        batch_time_m.update(time.time() - end)
        end = time.time()
        # measure elapsed time
        if i % 10 == 0:
            if device.index == 0:
                progress.display(i)
        # log train iter
        writer.add_scalar('train/iter_loss', loss.item(), train_iters+i)
        writer.add_scalar('train/iter_top1', acc1[0].item(), train_iters+i)

        # MetricCollector record
        metric_collector.record()
        metric_collector.place()

    # insert metrics and dump metrics
    if args.pyamp:
        precision = "amp"
    elif args.cnmix:
        precision = args.opt_level
    else:
        precision = "fp32"

    dev_cnt = dist.get_world_size() if args.rank == 0 else 1
    metric_collector.insert_metrics(
        net = "inceptionv2",
        batch_size = batch_size,
        precision = precision,
        cards = dev_cnt,
        DPF_mode = "ddp" if args.distributed else "single")

    if ((args.distributed == False) or (args.rank == 0)):
        metric_collector.dump()

    # log train epoch
    writer.add_scalar('train/epoch_loss', losses.avg, epoch)
    writer.add_scalar('train/epoch_top1', top1.avg, epoch)


def validate(val_loader, epoch, model, writer, device, iters=None):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [top1, top5],
        prefix='Test: [{}]'.format(epoch))

    # switch to evaluate mode
    model.eval()

    val_iters = len(val_loader) * epoch
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if iters>0 and i >= iters:
                break
            # set data to device
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = calcu_class_acc(output, target, topk=(1,5))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            if i % 10 == 0:
                if device.index == 0:
                    progress.display(i)
            # log valid iter
            writer.add_scalar('valid/iter_top1', acc1[0].item(), val_iters+i)

        if device.index == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))
        # log valid epoch
        writer.add_scalar('valid/epoch_top1', top1.avg, epoch)
        if device.index == 0:
            metric_collector = MetricCollector(enable_only_avglog=True)
            metric_collector.insert_metrics(net = "inceptionv2",
                                        accuracy = [top1.avg, top5.avg])
            metric_collector.dump()

    return top1.avg


if __name__ == '__main__':
    main()
