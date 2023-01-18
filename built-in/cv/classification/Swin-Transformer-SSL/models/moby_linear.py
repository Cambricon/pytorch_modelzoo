# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_pretrained, load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../tools/utils/")
from metric import MetricCollector

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=-1, #, required=True,
                        help='local rank for DistributedDataParallel, -1 stands for single card running.')
    # dev: linear eval settings
    parser.add_argument('--lr', type=float, default=1.0, help='the base lr for linear evaluation')
    parser.add_argument('--drop-path-rate', type=float, default=0.2, help='the drop path rate used in linear evaluation')
    # add new args
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu', 'mlu'],
                        help="device: cpu/gpu/mlu")
    parser.add_argument('--distributed', action='store_true',
                        help="use distributed data parallel.")
    parser.add_argument('--apex', action='store_true', default=False,
                        help='use pytorch apex for mixed precision training, currently only GPU are supported')
    parser.add_argument('--pyamp', action='store_true', default=False,
                        help='use pytorch amp for mixed precision training')
    parser.add_argument('--auto-resume', action='store_true', default=False,
                        help='auto resume from checkpoint')
    parser.add_argument('--iters', type=int, default=-1, metavar='N',
                        help='iters per epoch')
    parser.add_argument('--num-workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--pretrained-ckpt', help='pretrained checkpoint')
    parser.add_argument('--eval_iters', type=int, default=-1,
                        help='total eval iters for one epoch')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    config.defrost()
    # base
    config.LINEAR_EVAL.PRETRAINED = args.pretrained_ckpt
    config.OUTPUT = os.path.join(config.OUTPUT, 'linear')
    # model
    config.MODEL.TYPE = 'linear'
    config.MODEL.DROP_PATH_RATE = args.drop_path_rate
    # aug
    config.AUG.SSL_AUG = False
    config.AUG.SSL_LINEAR_AUG = True
    config.AUG.MIXUP = 0.0
    config.AUG.CUTMIX = 0.0
    config.AUG.CUTMIX_MINMAX = None
    # train
    config.TRAIN.EPOCHS = 100
    config.TRAIN.WARMUP_EPOCHS = 5
    # sched
    config.TRAIN.LR_SCHEDULER.NAME = 'cosine'
    # optim
    config.TRAIN.OPTIMIZER.NAME = 'sgd'
    config.TRAIN.OPTIMIZER.MOMENTUM = 0.9
    config.TRAIN.BASE_LR = args.lr
    config.TRAIN.WEIGHT_DECAY = 0.0
    config.freeze()

    return args, config

args, config = parse_option()

if args.device == 'gpu':
  try:
    import torch.backends.cudnn as cudnn
    # noinspection PyUnresolvedReferences
    from apex import amp
  except ImportError:
    amp = None
elif args.device == 'mlu':
  try:
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
  except ImportError:
    USE_MLU = None
    #raise ImportError("Import torch_mlu failed!")

from torch.cuda.amp import autocast, GradScaler

def main(config):
    _, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    if args.device == "gpu":
        model.cuda()
    elif args.device == "mlu":
        model = model.to("mlu")
    logger.info(str(model))

    # fix parameters except headmodel_without_ddp
    for name, p in model.named_parameters():
        if 'head' not in name:
            p.requires_grad = False

    optimizer = build_optimizer(config, model)
    # if MLU: the optimizer need to copy to device
    if args.device == "mlu":
        ct.to(optimizer, "mlu")

    scaler = None
    if args.pyamp:
        scaler = GradScaler()

    if args.apex and config.AMP_OPT_LEVEL != "O0":
        if args.device == "mlu":
            raise AssertionError("Swin transformer do not supported CNMIX, please run precision fp32 or AMP.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        model_without_ddp = model.module

        # load self-supervised pre-trained model
        load_pretrained(model_without_ddp, config.LINEAR_EVAL.PRETRAINED, logger)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")
        if hasattr(model_without_ddp, 'flops'):
            flops = model_without_ddp.flops()
            logger.info(f"number of GFLOPs: {flops / 1e9}")
    else:
        load_pretrained(model, config.LINEAR_EVAL.PRETRAINED, logger)

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        if args.distributed:
            max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, scaler, logger)
        else:
            max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, scaler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    # for internal benchmark test
    enable_only_benchmark = True if "BENCHMARK_LOG" in os.environ else False
    enable_only_avglog = True if "AVG_LOG" in os.environ else False
    metric_collector = MetricCollector(
            enable_only_benchmark=enable_only_benchmark,
            enable_only_avglog=enable_only_avglog,
            record_elapsed_time=True,
            record_hardware_time=True if args.device == "mlu" else False)

    logger.info("Start linear evaluation training")
    start_time = time.time()
    iters = 0
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        iters = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, scaler, iters, metric_collector)
        if args.distributed:
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, 0.0, optimizer, lr_scheduler, scaler, logger)
        else:
            if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
                save_checkpoint(config, epoch, model, 0.0, optimizer, lr_scheduler, scaler, logger)

        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        if args.iters == iters:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    if args.apex:
        precision = args.opt_level
    elif args.pyamp:
        precision = "amp"
    else:
        precision = "fp32"
    metric_collector.insert_metrics(
        net = "swin_transformer_ssl-Linear_evaluation",
        batch_size = args.batch_size,
        precision = precision,
        cards = int(os.environ['WORLD_SIZE']) if args.distributed else 1,
        DPF_mode = "ddp" if args.distributed else "single",
        accuracy=[round(acc1, 3), round(acc5, 3)])
    if (args.distributed and dist.get_rank() == 0) or args.distributed == False:
        metric_collector.dump()

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, scaler, iters, metric_collector):
    model.train()
    optimizer.zero_grad()
    real_run_steps = num_steps = len(data_loader)
    if args.iters > 0:
        real_run_steps = args.iters
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    metric_collector.place()
    for idx, (samples, targets) in enumerate(data_loader):
        if iters == args.iters:
            break
        iters += 1
        if args.device == "gpu":
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        elif args.device == "mlu":
            samples = samples.to("mlu", non_blocking=True)
            targets = targets.to("mlu", non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with autocast(enabled=args.pyamp):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if args.pyamp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                parameters = model.parameters()
            elif args.apex and config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                parameters = amp.master_params(optimizer)
            else:
                loss.backward()
                parameters = model.parameters()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(parameters)
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                if args.pyamp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            if args.pyamp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                parameters = model.parameters()
            elif args.apex and config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                parameters = amp.master_params(optimizer)
            else:
                loss.backward()
                parameters = model.parameters()

            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(parameters)
            if args.pyamp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        if args.distributed:
            if args.device == "gpu":
                torch.cuda.synchronize()
            elif args.device == "mlu":
                ct.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        # MetricCollector record
        metric_collector.record()
        metric_collector.place()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            if args.device == "gpu":
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            elif args.device == "mlu":
                memory_used = torch.mlu.max_memory_allocated() / (1024.0 * 1024.0)
            else:
                memory_used = 0.0
            etas = batch_time.avg * (real_run_steps - iters)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return iters


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        if idx == args.eval_iters:
            break
        if args.device == "gpu":
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        elif args.device == "mlu":
            images = images.to("mlu", non_blocking=True)
            target = target.to("mlu", non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if args.distributed:
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            if args.device == "gpu":
              memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            elif args.device == "mlu":
              memory_used = ct.max_memory_allocated() / (1024.0 * 1024.0)
            else:
              memory_used = 0.0
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        if args.device == "gpu":
            images = images.cuda(non_blocking=True)
        else:
            images = images.to(DEVICE, non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)

        if args.distributed:
            if args.device == "gpu":
                torch.cuda.synchronize()
            elif args.device == "mlu":
                ct.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
            config.defrost()
            config.LOCAL_RANK = rank
            config.freeze()
        else:
            rank = -1
            world_size = -1

        if args.device == "gpu":
            torch.cuda.set_device(config.LOCAL_RANK)
            torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
            torch.distributed.barrier()
        elif args.device == "mlu":
            ct.set_device(config.LOCAL_RANK)
            torch.distributed.init_process_group(backend='cncl', init_method='env://', world_size=world_size, rank=rank)
            torch.distributed.barrier()
        seed = config.SEED + dist.get_rank()
    else:
        seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.device == "gpu":
        cudnn.benchmark = True
        if torch.backends.cuda.matmul.allow_tf32 == True or torch.backends.cudnn.allow_tf32 == True:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print("TF32 is modified False, start FP32 Running")
    if args.distributed:
        # linear scale the learning rate according to total batch size, may not be optimal
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    else:
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    if args.distributed:
        logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    else:
        logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    if args.distributed:
        if dist.get_rank() == 0:
            path = os.path.join(config.OUTPUT, "config.json")
            with open(path, "w") as f:
                f.write(config.dump())
            logger.info(f"Full config saved to {path}")
    else:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
