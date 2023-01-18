# -*- coding: utf-8 -*-

'''
Janurary 2018 by Wei Li
liweihfyz@sjtu.edu.cn
https://www.github.cim/leviswind/transformer-pytorch
'''
from __future__ import print_function

import argparse
import random
import os
import time
import re
import logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from AttModel import AttModel
from data_load import TrainDataSet, load_de_vocab, load_en_vocab
from hyperparams import Hyperparams as hp
from util import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../tools/utils/")
from metric import MetricCollector

try:
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
except ImportError:
    print("import torch_mlu failed!")

try:
    import cnmix
except ImportError:
    print("train without cnmix")

logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("__name__")

def main(args):
    if args.device == "CPU" and args.distributed:
        print("The CPU device platform does not support distributed operation.")
        return

    if not os.path.exists(args.ckp_path):
        os.makedirs(args.ckp_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    if args.device == "MLU":
        ndev_per_node = ct.device_count()
    if args.device == "GPU":
        ndev_per_node = torch.cuda.device_count()
    args.world_size = ndev_per_node * args.world_size
    import time
    start = time.time()
    if args.distributed:
        if (sys.version_info[0] < 3):
            if not os.getenv('WORLD_SIZE') or not os.getenv('LOCAL_RANK'):
                print("WORLD_SIZE or LOCAL_RANK is empty!")
                sys.exit(1)
            main_worker(os.getenv('LOCAL_RANK'), os.getenv('WORLD_SIZE'), args)
        else:
            if not os.getenv('MASTER_ADDR'):
                os.environ['MASTER_ADDR'] = args.master_addr
            if not os.getenv('MASTER_PORT'):
                os.environ['MASTER_PORT'] = args.master_port
            mp.spawn(main_worker, nprocs=ndev_per_node, args=(ndev_per_node, args, ), join=True)
    else:
        main_worker(-1, ndev_per_node, args)
    end = time.time()
    print("Using Time: " + str(end-start))


def main_worker(rank, ndev_per_node, args):
    if args.seed is not None:
        set_seed(args.seed)

    rank = (int)(rank)
    ndev_per_node = (int)(ndev_per_node)
    if args.device == "MLU":
        ct.set_device(0 if rank == -1 else rank)
    if args.device == "GPU":
        torch.cuda.set_device(0 if rank == -1 else rank)
    # distributed training env setting up
    if args.distributed:
        dist.init_process_group(backend='cncl' if args.device == "MLU" else 'nccl', rank=rank, world_size=ndev_per_node)

    startepoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location='cpu')
        startepoch = state['epoch'] + 1

    # Load data
    source_train = args.dataset_path + hp.source_train
    target_train = args.dataset_path + hp.target_train
    train_dataset =  TrainDataSet(source_train, target_train)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas = ndev_per_node, rank = rank)
        if os.getenv('BENCHMARK_LOG') is None:
            args.batch_size = args.batch_size // ndev_per_node
        else:
            args.lr = args.lr * ndev_per_node
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size= args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler)

    hp.dropout_rate = args.dropout_rate

    # distributed model
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    enc_voc = len(de2idx)
    dec_voc = len(en2idx)
    model = AttModel(hp, enc_voc, dec_voc)

    # adaptive_quantize
    if args.device == "MLU":
        model.to(ct.mlu_device())
    if args.device == "GPU":
        model.cuda()

    # load state_dict
    if args.resume:
        model.load_state_dict(state['model'], strict=False)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr, betas=[0.9, 0.98], eps=1e-8)
    if args.device == "MLU"  and args.cnmix:
        model, optimizer = cnmix.initialize(model, optimizer, opt_level = args.opt_level)
        cnmix.cnmix_set_amp_quantify_params('all', {'batch_size': args.batch_size,
                                                     'data_num': args.batch_size * len(train_loader)})
        if args.resume and isinstance(state, dict) and 'cnmix' in state:
            cnmix.load_state_dict(state['cnmix'])

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0 if rank == -1 else rank])

    if args.resume:
        optimizer.load_state_dict(state['optim'])

        if args.device == "MLU":
            ct.to(optimizer, torch.device('mlu'))

    if args.device == "GPU":
        cudnn.benckmark = True

    if args.seed is not None:
        set_seed(args.seed)

    for epoch in range(startepoch, args.num_epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        epoch_log = os.path.join(args.log_path, "epoch{:02d}_rank{:02d}.txt".format(epoch, -1))

        epoch_iters = len(train_loader)

        train(train_loader, model, optimizer, epoch, args, epoch_log, rank, epoch_iters)

        # save model
        if not args.save_ckpt:
            break
        if not args.distributed or ( args.distributed and rank % ndev_per_node == 0 ):
            checkpoint_path = os.path.join(args.ckp_path, "model_epoch_{:02d}.pth".format(epoch))
            state = {}
            state['epoch'] = epoch
            if args.distributed:
                state['model'] = model.module.state_dict()
            else:
                state['model'] = model.state_dict()
            state['optim'] = optimizer.state_dict()
            if args.cnmix:
                state['cnmix']=cnmix.state_dict()
            torch.save(state, checkpoint_path)

    if args.distributed:
        dist.destroy_process_group()

def set_lr(optimizer, args, cur_iters):
    if cur_iters <= args.warmup_iters:
        lr = float(args.lr * cur_iters) / float(args.warmup_iters)
    else:
        lr = float(args.lr * args.warmup_iters**0.5 * cur_iters ** -0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, optimizer, epoch, args, epoch_log, rank, epoch_iters):
    adaptive_cnt = int(os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT')) if (
            os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT') is not None) else 0
    batch_time_benchmark = []
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    losses = AverageMeter('Loss', ':.4e')
    acces = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acces],
        prefix="Card{} Epoch: [{}]".format(rank, epoch))

    model.train()

    if args.device == "GPU":
        torch.cuda.synchronize()
    end = time.time()

    cur_iters = (epoch - 1) * epoch_iters + 1

    # for internal benchmark test
    metric_collector = MetricCollector(
        enable_only_benchmark=True,
        record_elapsed_time=True,
        record_hardware_time=True if args.device == 'MLU' else False)
    metric_collector.place()

    for i, (data, target) in enumerate(train_loader):
        set_lr(optimizer, args, cur_iters)
        if (i == args.iterations):
            logger.info('The program iteration runs out. iterations: %d' % args.iterations)
            break

        data_time.update(time.time() - end)
        if args.device == "GPU":
            data = data.cuda()
            target = target.cuda()
        if args.device == "MLU":
            data = data.to(ct.mlu_device(), non_blocking=True)
            target = target.to(ct.mlu_device(), non_blocking=True)

        loss, _, acc = model(data, target)
        losses.update(loss.item())
        acces.update(acc.item())

        optimizer.zero_grad()
        if args.device == 'MLU' and args.cnmix:
            with cnmix.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        if args.device == "GPU":
            torch.cuda.synchronize()
        
        # MetricCollector record
        metric_collector.record()
        metric_collector.place()

        batch_time.update(time.time() - end)
        end = time.time()

        if rank <= 0:
           msglog(epoch_log, "{}, {}".format(loss.item(), acc.item()))
        if i % args.print_freq == 0:
            progress.display(i)
    if args.device == "MLU":
        cards = ct.device_count() if rank == 0 else 1
    if args.device == "GPU":
        cards = torch.cuda.device_count() if rank == 0 else 1
    if args.cnmix:
        precision = args.opt_level
    else:
        precision = "fp32"

    metrics = metric_collector.get_metrics()
    if 'batch_time_avg' in metrics:
        metric_collector.insert_metrics(
            throughput = args.batch_size * hp.maxlen / metrics['batch_time_avg'] * cards)
    metric_collector.insert_metrics(
        net = "transformer",
        batch_size = args.batch_size,
        precision = precision,
        cards = cards,
        DPF_mode = "ddp " if args.distributed == True else "single")
    if ((args.distributed == False) or (rank == 0)):
        metric_collector.dump()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer training.")
    parser.add_argument('--seed', default=66, type=int, help='random seed.')
    parser.add_argument('--log-path', default='logs', type=str, help='training log path.')
    parser.add_argument('--ckp-path', default='models', type=str, help='training ckps path.')
    parser.add_argument('--resume', type=str, help='resume ckp path.')
    parser.add_argument('--batch-size', default=32, type=int, help='training batch size for all')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers.')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training.')
    parser.add_argument('--rank', default=0, type=int, help='node rank fro distributed training.')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency of information.')
    parser.add_argument('--distributed', action='store_true', help='distributed training.')
    parser.add_argument('--save_ckpt', default=True, type=bool, help='save checkpoint.')
    parser.add_argument('--device', default='MLU', type=str, help='set the type of hardware used for training.')
    parser.add_argument('--bitwidth', default=8, type=int, help="Set the initial quantization width of network training.")
    parser.add_argument('--iterations', default=-1, type=int, help="Number of training iterations.")
    parser.add_argument('--dataset-path', default='corpora/', type=str, help='The path of imagenet dataset.')
    parser.add_argument('--num_epochs', default=20, type=int, help='Number of training num_epochs.')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate.')
    parser.add_argument('--master-addr', default='127.0.0.1', type=str, help='ddp address.')
    parser.add_argument('--master-port', default='29501', type=str, help='ddp address port.')
    parser.add_argument('--warmup-iters', default=300, type=float, help='warm up iterations')
    parser.add_argument('--lr', "--learning-rate", default=0.0005, type=float, help="learning rate for training")
    parser.add_argument('--cnmix', action='store_true', default=False, help='use cnmix for mixed precision training')
    parser.add_argument('--opt_level', type=str, default="O0", help='choose level of mixing precision')
    args = parser.parse_args()
    main(args)

