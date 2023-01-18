# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from builtins import print
from calendar import c
import os
import sys
import time
from argparse import ArgumentParser
import torch
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data.distributed
from torch.cuda.amp import autocast, GradScaler

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../tools/utils/")
sys.path.append(cur_dir + "/models")
from metric import MetricCollector

from model import SSD300, ResNet, Loss
from utils import dboxes300_coco, Encoder
from logger import Logger, BenchLogger
from evaluate import evaluate

from train import train_loop, tencent_trick, load_checkpoint, benchmark_train_loop, benchmark_inference_loop
from data import get_train_loader, get_val_dataset, get_val_dataloader, get_coco_ground_truth

import dllogger as DLLogger
import random

cur_dir = os.path.dirname(os.path.abspath(__file__))

def generate_mean_std(args):
    mean_val = [0.485, 0.456, 0.406]
    std_val = [0.229, 0.224, 0.225]

    if args.device == "GPU":
        mean = torch.tensor(mean_val).cuda()
        std = torch.tensor(std_val).cuda()
    if args.device == "MLU":
        mean = torch.tensor(mean_val).mlu()
        std = torch.tensor(std_val).mlu()

    view = [1, len(mean_val), 1, 1]

    mean = mean.view(*view)
    std = std.view(*view)

    return mean, std

def make_parser():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--device', type=str, default='MLU',
                        help='set the type of hardware used for training.')
    parser.add_argument('--data', '-d', type=str, default='/coco', required=True,
                        help='path to test and training data files')
    parser.add_argument('--epochs', '-e', type=int, default=65,
                        help='number of epochs for training')
    parser.add_argument('--batch-size', '--bs', type=int, default=32,
                        help='number of examples for each iteration')
    parser.add_argument('--eval-batch-size', '--ebs', type=int, default=32,
                        help='number of examples for each evaluation iteration')
    parser.add_argument('--no-mlu', action='store_true',
                        help='use available MLUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint file')
    parser.add_argument('--save', type=str, default=None,
                        help='save model checkpoints in the specified directory')
    parser.add_argument('--mode', type=str, default='training',
                        choices=['training', 'evaluation', 'benchmark-training', 'benchmark-inference'])
    parser.add_argument('--evaluation', nargs='*', type=int, default=[],
                        # default=[21, 31, 37, 42, 48, 53, 59, 64],
                        help='epochs at which to evaluate')
    parser.add_argument('--multistep', nargs='*', type=int, default=[43, 54],
                        help='epochs at which to decay learning rate')
    parser.add_argument('--start_eval_at', dest='start_eval_at', type=int, default=None, 
                                help='start evaluation at specified epoch')
    parser.add_argument('--evaluate_every', '--eval_every', dest='evaluate_every', type=int, default=None,
                                help='evaluate at every epochs')
    parser.add_argument('--target_map', dest='target_map', type=float, default=None, 
                                help='target map')
    parser.add_argument("--data-backend",
                        metavar="BACKEND",
                        default="pytorch",
                        help="data backend: "
                        + "pytorch, dali-mlu"
                        + " (default: pytorch)",)

    # Hyperparameters
    parser.add_argument('--learning-rate', '--lr', type=float, default=2.6e-3,
                        help='learning rate')
    parser.add_argument('--momentum', '-m', type=float, default=0.9,
                        help='momentum argument for SGD optimizer')
    parser.add_argument('--weight-decay', '--wd', type=float, default=0.0005,
                        help='momentum argument for SGD optimizer')

    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--iterations', type=int, default=20, metavar='N',
                        help='Run N iterations while training (ignored when benchmark)')

    parser.add_argument('--eval_iters', type=int, default=-1, metavar='N',
                        help='Run N iterations while evaluate (ignored when benchmark)')
    parser.add_argument('--benchmark-iterations', type=int, default=20, metavar='N',
                        help='Run N iterations while benchmarking (ignored when training and validation)')
    parser.add_argument('--benchmark-warmup', type=int, default=20, metavar='N',
                        help='Number of warmup iterations for benchmarking')

    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--backbone-path', type=str, default=None,
                        help='Path to chekcpointed backbone. It should match the'
                             ' backbone model declared with the --backbone argument.'
                             ' When it is not provided, pretrained model from torchvision'
                             ' will be downloaded.')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--json-summary', type=str, default=None,
                        help='If provided, the json summary will be written to'
                             'the specified file.')

    # Distributed
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK',0), type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')

    parser.add_argument('--max_bitwidth',
                        action='store_true',
                        help='use Max Bitwidth of MLU training.')

    parser.add_argument('--cnmix', action='store_true',
                        help='use cnmix')

    parser.add_argument('--pyamp', action='store_true', default=False,
                    help='use pytorch amp for mixed precision training')

    parser.add_argument('--opt_level', type=str, default='O0',
                        help='cnmix optimizer level')

    return parser


def train(train_loop_func, logger, args):
    # Setup multi-device if necessary
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        print('Using DDP')
        if args.device == "MLU":
            ct.set_device(args.local_rank)
        elif args.device == "GPU":
            torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='cncl' if args.device == "MLU" else 'nccl', rank=args.local_rank, world_size=int(os.environ['WORLD_SIZE']))
        args.N_gpu = torch.distributed.get_world_size()
    else:
        args.N_gpu = 1

    if args.seed is None:
        args.seed = np.random.randint(1e4)

    if args.distributed:
        args.seed = (args.seed + torch.distributed.get_rank()) % 2**32
    print("Using seed = {}".format(args.seed))
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    # Setup data, defaults
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    cocoGt = get_coco_ground_truth(args)

    train_loader = get_train_loader(args, args.seed - 2**31)

    val_dataset = get_val_dataset(args)
    val_dataloader = get_val_dataloader(val_dataset, args)

    ssd300 = SSD300(backbone=ResNet(args.backbone, args.backbone_path))
    args.learning_rate = args.learning_rate * args.N_gpu * (args.batch_size / 32)
    start_epoch = 0
    iteration = 0
    loss_func = Loss(dboxes)

    ssd300.train()
    # model to mlu
    if args.device == "MLU":
        ssd300.to('mlu')
        loss_func.to('mlu')
    elif args.device == "GPU":
        ssd300.cuda()
        loss_func.cuda()
    
    if args.pyamp:
        scaler = GradScaler()
    else:
        scaler = None

    optimizer = torch.optim.SGD(tencent_trick(ssd300), lr=args.learning_rate,
                                    momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            load_checkpoint(ssd300, args.checkpoint)
            checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.pyamp and isinstance(checkpoint, dict) and 'amp' in checkpoint:
                    scaler.load_state_dict(checkpoint['amp'])
        else:
            print('Provided checkpoint is not path to a file')
            return

    if args.device == "MLU" and getattr(args, 'cnmix', False):
        import cnmix
        ssd300, optimizer=cnmix.initialize(ssd300, optimizer, opt_level=args.opt_level)
        cnmix.core.cnmix_set_amp_quantify_params('all',{'batch_size': args.batch_size,
                                                        'data_num': args.batch_size * len(train_loader)})
        if args.checkpoint is not None and os.path.isfile(args.checkpoint):
            if isinstance(checkpoint, dict) and 'cnmix' in checkpoint:
                cnmix.load_state_dict(['cnmix'])

    if args.device == "MLU":
        ct.to(optimizer, torch.device('mlu'))
    if args.distributed:
        ssd300 = torch.nn.parallel.DistributedDataParallel(ssd300, device_ids=[args.local_rank])

    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    inv_map = {v: k for k, v in val_dataset.label_map.items()}

    total_time = 0

    if args.mode == 'evaluation':
        print("Evaluating...")
        acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)
        if args.local_rank == 0:
            print('Model precision {} mAP'.format(acc))
            metric_collector = MetricCollector(enable_only_avglog=True)
            metric_collector.insert_metrics(net = "SSD_ResNet50",
                                        accuracy = acc)
            metric_collector.dump()

        return
    mean, std = generate_mean_std(args)

    next_eval_at = args.start_eval_at

    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()
        iteration = train_loop_func(ssd300, loss_func, epoch, optimizer, train_loader, val_dataloader, encoder, iteration,
                                    logger, args, mean, std, scaler=scaler)
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time
        
        if args.local_rank == 0:
            logger.update_epoch_time(epoch, end_epoch_time)

        if epoch in args.evaluation:
            acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)

            if args.local_rank == 0:
                logger.update_epoch(epoch, acc)

        if args.save and args.local_rank == 0:
            obj = {'epoch': epoch + 1,
                   'iteration': iteration,
                   'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(),
                   'label_map': val_dataset.label_info}
            if args.distributed:
                obj['model'] = ssd300.module.state_dict()
            else:
                obj['model'] = ssd300.state_dict()
            if args.pyamp:
                obj['amp'] = scaler.state_dict()
            if not args.no_mlu and getattr(args, 'cnmix', False):
                import cnmix
                obj['cnmix'] = cnmix.state_dict()
            if epoch == args.epochs - 1:
                print("saving model...")
                save_path = os.path.join(args.save, f'last.pt') 
                torch.save(obj, save_path)
                logger.log('model path', save_path)
        if args.target_map is not None:
            if epoch == next_eval_at:
                print("evaluationg started")
                args.distributed = False
                acc = None
                val_dataloader = get_val_dataloader(val_dataset, args)
                next_eval_at += args.evaluate_every
                if args.local_rank == 0:
                    acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)
                if 'WORLD_SIZE' in os.environ:
                    args.distributed = int(os.environ['WORLD_SIZE']) > 1
                if acc is not None and acc >= args.target_map:
                    print("{} map achieved, training finished".format(acc))
                    break
        # UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`.
        # In PyTorch 1.1.0 and later, you should call them in the opposite order:
        # `optimizer.step()` before `lr_scheduler.step()`.
        # Failure to do this will result in PyTorch skipping the first value of the
        # learning rate schedule.See more details at
        #  https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        scheduler.step()
        if args.data_backend != "pytorch":
            train_loader.reset()
            
    DLLogger.log((), { 'total time': total_time })
    logger.log_summary()

def log_params(logger, args):
    logger.log_params({
        "dataset path": args.data,
        "epochs": args.epochs,
        "batch size": args.batch_size,
        "eval batch size": args.eval_batch_size,
        "device": args.device,
        "seed": args.seed,
        "checkpoint path": args.checkpoint,
        "mode": args.mode,
        "eval on epochs": args.evaluation,
        "lr decay epochs": args.multistep,
        "learning rate": args.learning_rate,
        "momentum": args.momentum,
        "weight decay": args.weight_decay,
        "lr warmup": args.warmup,
        "iterations": args.iterations,
        "backbone": args.backbone,
        "backbone path": args.backbone_path,
        "num workers": args.num_workers
    })

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    if args.device == "MLU":
        import torch_mlu
        import torch_mlu.core.mlu_model as ct
        global ct
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if args.local_rank == 0:
        os.makedirs('./models', exist_ok=True)

    torch.backends.cudnn.benchmark = True

    # write json only on the main thread
    args.json_summary = args.json_summary if args.local_rank == 0 else None

    if args.mode == 'benchmark-training':
        train_loop_func = benchmark_train_loop
        logger = BenchLogger('Training benchmark', json_output=args.json_summary)
        args.epochs = 1
    elif args.mode == 'benchmark-inference':
        train_loop_func = benchmark_inference_loop
        logger = BenchLogger('Inference benchmark', json_output=args.json_summary)
        args.epochs = 1
    else:
        train_loop_func = train_loop
        logger = Logger('Training logger', print_freq=1, json_output=args.json_summary)

    log_params(logger, args)

    train(train_loop_func, logger, args)
