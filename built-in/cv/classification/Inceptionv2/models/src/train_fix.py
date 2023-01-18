from quanz import *
import argparse
import math
import os
import random
import shutil


import numpy as np
import torch
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter


from utils import *

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
parser.add_argument('--rank', default=None, type=int,
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

# quanz parameters
parser.add_argument('--second-stage-iters', default=None, type=int,
                    help='Adaptive training, second stage iterations.')
parser.add_argument('--max-update-iters', default=None, type=int,
                    help='Adaptive traingin, maximum update iterations.')
parser.add_argument('--input-init-bit', default=None, type=int,
                    help='Input initial bit.')
parser.add_argument('--input-mode', default=None, type=str, choices=train_modes,
                    help='Input training mode.')
parser.add_argument('--weight-init-bit', default=None, type=int,
                    help='Weight initial bit.')
parser.add_argument('--weight-mode', default=None, type=str, choices=train_modes,
                    help='Weight training mode.')
parser.add_argument('--grad-init-bit', default=None, type=int,
                    help='Grad initial bit.')
parser.add_argument('--grad-mode', default=None, type=str, choices=train_modes,
                    help='Grad training mode.')
parser.add_argument('--infer', default=None, type=str, choices=infer_modes,
                    help='Infer mode.')

# checkin parameters
parser.add_argument('--daily', action='store_true',
                    help='daily check in.')

def main():
    # load arguments and configs
    args = parser.parse_args()
    config = precess_train_fix_config(args)
    # init seed
    init_seed(args.seed, not args.use_cpu)
    
    # prepare 
    if args.use_cpu: # not support in cpu mode
        args.distribued = False
    else:
        ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        config['train']['world_size'] = ngpus_per_node * config['train']['world_size']
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config))
    else:
        main_worker(args.gpu, ngpus_per_node, args, config)


def init_quanz(config, rank, total, prefix='runs'):
    param.debug = True
    param.second_stage_iters = config['second_stage_iters']
    param.max_update_iters = config['max_update_iters']
    param.infer = config['infer']
    if param.debug:
        param.writer = set_writer(prefix+"_quanz", rank, total)
    for key in param.update_thred.keys():
        param.input_bit[key] = config['input_init_bit']
        param.weight_bit[key] = config['weight_init_bit']
        param.grad_bit[key] = config['grad_init_bit']
        mode.input_mode[key] = config['input_mode']
        mode.weight_mode[key] = config['weight_mode']
        mode.grad_mode[key] = config['grad_mode']

def main_worker(gpu, ngpus_per_node, args, config):
    # prepare device 
    args.gpu = gpu
    device = set_device(args.gpu, args.use_cpu)

    # init envs
    prefix = 'daily_fix' if args.daily else 'logs_fix'
    prefix = "{}_{}".format(prefix, config['model'])
    writer = set_writer(prefix, args.gpu, ngpus_per_node)
    init_quanz(config['quanz'], args.gpu, ngpus_per_node, prefix)
    if args.distributed:
        init_dist(config['train'], args.gpu, ngpus_per_node)

    # create model by default is not pretrained model
    model = create_model(
        config['model'], args.pretrained, config['pretrain']['path']).to(device)

    # Data load
    train_loader, train_sampler = create_loader(config['train_dataset'], config['train'], args.distributed)
    val_loader, _ = create_loader(config['valid_dataset'], config['valid'])

    # create criterion, optimizer, scheduler  
    criterion = create_criterion(config['train']['criterion']).to(device)
    optimizer = create_optimizer(model, config['train']['optimizer'])
    scheduler = Scheduler(config['train']['scheduler'], config['train']['optimizer']['learning_rate'], len(train_loader))

    # optionally resume from a checkpoint
    if args.resume:
        modules = dict()
        for key in param.update_thred.keys():
            modules[key] = []

        get_quanz_module(model, modules)
        init_quanz_module(modules, device)

        args.start_epoch, best_acc1 = resume_train(args.resume, model, optimizer, device)
    else:
        best_acc1 = 0.0
    if args.distributed:
        model = convert_ddp_model(model, config['train']['rank'])

    # daily checkin
    if args.daily:
        daily_iters = 1000
        args.ckpt_path = prefix
    else:
        daily_iters = None

    for epoch in range(args.start_epoch, config['train']['scheduler']['epochs']):
        # random sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, epoch,  model, criterion, optimizer, scheduler, writer, device, daily_iters)
        
        # evaluate on validation set
        acc1_val = validate(val_loader, epoch, model, writer, device)

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
            if ((epoch+1) % (config['train']['scheduler']['epochs'] // 10)) == 0:
                save_ckpt(state, False, path='{}_{}'.format(args.ckpt_path, config['model']), epoch=epoch)
            save_ckpt(state, is_best, path='{}_{}'.format(args.ckpt_path, config['model']))
        if args.daily:
            break
    
    if args.gpu is None or args.gpu == 0:
        bit_num = calcu_bit(model)
        print(bit_num)
        
    writer.close()      
    if param.debug:
        param.writer.close()
 

def train(train_loader, epoch, model, criterion, optimizer, scheduler, writer, device, iters=None):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1],
        prefix="Train: [{}]".format(epoch))

    # switch to train mode 
    model.train()
    
    train_iters = len(train_loader) * epoch
    for i, (images, target) in enumerate(train_loader):
        
        # for daily checkin
        if iters and i >= iters:
            return
        # adjust leaning rate
        adjust_lr(optimizer, scheduler.get_lr(epoch, train_iters+i))
        writer.add_scalar('train/lr', scheduler.cur_lr, train_iters+i)

        # set data to device
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = calcu_class_acc(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        if i % 10 == 0:
            if device.index == 0:
                progress.display(i)
        # log train iter
        writer.add_scalar('train/iter_loss', loss.item(), train_iters+i)
        writer.add_scalar('train/iter_top1', acc1[0].item(), train_iters+i)
    # log train epoch
    writer.add_scalar('train/epoch_loss', losses.avg, epoch)
    writer.add_scalar('train/epoch_top1', top1.avg, epoch)


def validate(val_loader, epoch, model, writer, device):
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [top1],
        prefix='Test: [{}]'.format(epoch))

    # switch to evaluate mode
    model.eval()

    val_iters = len(val_loader) * epoch
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            # set data to device
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)

            # measure accuracy and record loss
            acc1 = calcu_class_acc(output, target, topk=(1,))
            top1.update(acc1[0].item(), images.size(0))
            
            if i % 10 == 0:
                if device.index == 0: 
                    progress.display(i)
            # log valid iter
            writer.add_scalar('valid/iter_top1', acc1[0].item(), val_iters+i)

        if device.index == 0:
            print(' * Acc@1 {top1.avg:.3f}'
                .format(top1=top1))
        # log valid epoch
        writer.add_scalar('valid/epoch_top1', top1.avg, epoch)

    return top1.avg


if __name__ == '__main__':
    main()
