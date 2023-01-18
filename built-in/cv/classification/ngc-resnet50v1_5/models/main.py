# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import argparse
import os
import shutil
import time
import re
import random

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../tools/utils/")
from metric import MetricCollector

try:
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
except ImportError:
    print("Import torch_mlu failed!\n")

try:
    import cnmix
except ImportError:
    print("import cnmix failed!\n")

import image_classification.resnet as models
import image_classification.logger as log

from image_classification.smoothing import LabelSmoothing
from image_classification.mixup import NLLMultiLabelSmooth, MixUpWrapper
from image_classification.dataloaders import *
from image_classification.training import *
from image_classification.utils import *

import dllogger


def add_parser_arguments(parser):
    model_names = models.resnet_versions.keys()
    model_configs = models.resnet_configs.keys()

    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "--data-backend",
        metavar="BACKEND",
        default="pytorch",
        help="data backend choices are pytorch, dali-gpu, dali-mlu",
    )

    parser.add_argument(
        "--interpolation",
        metavar="INTERPOLATION",
        default="bilinear",
        help="interpolation type for resizing images: bilinear, bicubic or triangular(DALI only)",
    )

    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="resnet50",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
    )

    parser.add_argument(
        "--model-config",
        "-c",
        metavar="CONF",
        default="classic",
        choices=model_configs,
        help="model configs: " + " | ".join(model_configs) + "(default: classic)",
    )

    parser.add_argument(
        "--num-classes",
        metavar="N",
        default=1000,
        type=int,
        help="number of classes in the dataset",
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=5,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--run-epochs",
        default=-1,
        type=int,
        metavar="N",
        help="run only N epochs, used for checkpointing runs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256) per gpu",
    )

    parser.add_argument(
        "--optimizer-batch-size",
        default=-1,
        type=int,
        metavar="N",
        help="size of a total batch size, for simulating bigger batches using gradient accumulation",
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--lr-schedule",
        default="step",
        type=str,
        metavar="SCHEDULE",
        choices=["step", "linear", "cosine"],
        help="Type of LR schedule: {}, {}, {}".format("step", "linear", "cosine"),
    )

    parser.add_argument(
        "--warmup", default=0, type=int, metavar="E", help="number of warmup epochs"
    )

    parser.add_argument(
        "--label-smoothing",
        default=0.0,
        type=float,
        metavar="S",
        help="label smoothing",
    )
    parser.add_argument(
        "--mixup", default=0.0, type=float, metavar="ALPHA", help="mixup alpha"
    )

    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--bn-weight-decay",
        action="store_true",
        help="use weight_decay on batch normalization learnable parameters, (default: false)",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        help="use nesterov momentum, (default: false)",
    )

    parser.add_argument(
        "--print-freq",
        "-p",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--pretrained-weights",
        default="",
        type=str,
        metavar="PATH",
        help="load weights from here",
    )

    parser.add_argument("--fp16", action="store_true", help="Run model fp16 mode.")
    parser.add_argument(
        "--static-loss-scale",
        type=float,
        default=1,
        help="Static loss scale, positive power of 2 values can improve fp16 convergence.",
    )
    parser.add_argument(
        "--dynamic-loss-scale",
        action="store_true",
        help="Use dynamic loss scaling.  If supplied, this argument supersedes "
        + "--static-loss-scale.",
    )
    parser.add_argument(
        "--prof", type=int, default=-1, metavar="N", help="Run only N iterations"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run model AMP (automatic mixed precision) mode.",
    )
    parser.add_argument(
        "--cnmix",
        action="store_true",
        help="Use cnmix.",
    )
    parser.add_argument(
        "--opt_level",
        type=str,
        default="O1",
        help="Run model with O1,O2,O3,O4 mode.",
    )

    parser.add_argument(
        "--seed", default=None, type=int, help="random seed used for numpy and pytorch"
    )

    parser.add_argument(
        "--gather-checkpoints",
        action="store_true",
        help="Gather checkpoints throughout the training, without this flag only best and last checkpoints will be stored",
    )

    parser.add_argument(
        "--raport-file",
        default="experiment_raport.json",
        type=str,
        help="file in which to store JSON experiment raport",
    )

    parser.add_argument(
        "--evaluate", action="store_true", help="evaluate checkpoint/model"
    )
    parser.add_argument("--training-only", action="store_true", help="do not evaluate")

    parser.add_argument(
        "--no-checkpoints",
        action="store_false",
        dest="save_checkpoints",
        help="do not store any checkpoints, useful for benchmarking",
    )

    parser.add_argument("--checkpoint-filename", default="checkpoint.pth.tar", type=str)

    parser.add_argument(
        "--workspace",
        type=str,
        default="./",
        metavar="DIR",
        help="path to directory where checkpoints will be stored",
    )
    parser.add_argument(
        "--memory-format",
        type=str,
        default="nchw",
        choices=["nchw", "nhwc"],
        help="memory layout, nchw or nhwc",
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=-1,
        metavar='N',
        help="run N iters while training",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="mlu",
        choices=["mlu","gpu","cpu"],
        help="device to run the model",
    )
    parser.add_argument(
        '--dist-backend',
        default='nccl',
        type=str,
        help='distributed backend')
    parser.add_argument(
        '--dummy_test',
        dest='dummy_test',
        action='store_true',
        help='use fake data to training')
    parser.add_argument(
        '--pyamp',
        dest='pyamp',
        action='store_true',
        help='use torch amp to training')

def calc_ips(batch_size, time):
    world_size = (
            torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    tbs = world_size * batch_size
    return tbs / time

def remove_module_prefix(state):
    state_keys = list(state.keys())
    for key in state_keys:
        if key.startswith("module."):
            state[key[7:]] = state[key]
            state.pop(key)

def main(args):
    exp_start_time = time.time()
    global best_prec1
    best_prec1 = 0
    global device_global
    device_global = args.device

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.local_rank = int(os.environ["LOCAL_RANK"])

    args.mlu = 0
    args.world_size = 1

    if args.distributed:
        if args.device == "mlu":
            args.mlu = args.local_rank % ct.device_count()
            ct.set_device(args.mlu)
        else:
            args.gpu = args.local_rank % torch.cuda.device_count()
            torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend=args.dist_backend, init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    if args.amp and args.fp16:
        print("Please use only one of the --fp16/--amp flags")
        exit(1)

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)
        random.seed(args.seed)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + id)
            random.seed(args.seed + id)

    else:

        def _worker_init_fn(id):
            pass

    if args.fp16:
        assert (
            torch.backends.cudnn.enabled
        ), "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = args.world_size * args.batch_size
        if args.optimizer_batch_size % tbs != 0:
            print(
                "Warning: simulated batch size {} is not divisible by actual batch size {}".format(
                    args.optimizer_batch_size, tbs
                )
            )
        batch_size_multiplier = int(args.optimizer_batch_size / tbs)
        print("BSM: {}".format(batch_size_multiplier))

    pretrained_weights = None
    if args.pretrained_weights:
        if os.path.isfile(args.pretrained_weights):
            print(
                "=> loading pretrained weights from '{}'".format(
                    args.pretrained_weights
                )
            )
            pretrained_weights = torch.load(args.pretrained_weights)
            # Temporary fix to allow NGC checkpoint loading
            pretrained_weights = {
                k.replace("module.", ""): v for k, v in pretrained_weights.items()
            }
        else:
            print("=> no pretrained weights found at '{}'".format(args.resume))

    start_epoch = 0

    scaler = None
    if args.pyamp:
        scaler = GradScaler()

    # optionally resume from a checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=torch.device('cpu')
            )
            start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model_state = checkpoint["state_dict"]
            remove_module_prefix(model_state)
            optimizer_state = checkpoint["optimizer"]
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            if args.pyamp:
                if isinstance(checkpoint, dict) and 'amp' in checkpoint:
                    scaler.load_state_dict(checkpoint['amp'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            model_state = None
            optimizer_state = None
    else:
        model_state = None
        optimizer_state = None

    loss = nn.CrossEntropyLoss
    if args.mixup > 0.0:
        loss = lambda: NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        loss = lambda: LabelSmoothing(args.label_smoothing)

    memory_format = (
        torch.channels_last if args.memory_format == "nhwc" else torch.contiguous_format
    )

    model_and_loss = ModelAndLoss(
        (args.arch, args.model_config, args.num_classes),
        loss,
        pretrained_weights=pretrained_weights,
        device=args.device,
        fp16=args.fp16,
        memory_format=memory_format,
    )

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    train_dataset = datasets.ImageFolder(traindir,
                             transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 normalize,]))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if args.data_backend == "pytorch":
        train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.workers,
                pin_memory=True)
        train_loader_len = len(train_loader)
    elif args.data_backend == "dali-mlu":
        from image_classification.dataloaders_dali import get_dali_train_loader
        get_train_loader = get_dali_train_loader(dali_cpu=False)
        train_loader, train_loader_len = get_train_loader(
                                             args.data,
                                             224,#image_size,
                                             args.batch_size,
                                             args.num_classes,
                                             args.mixup > 0.0,
                                             interpolation=args.interpolation,
                                             start_epoch=start_epoch,
                                             workers=args.workers,
                                         )
    elif args.data_backend == "dali-gpu":
        from image_classification.dataloaders_dali import get_dali_train_loader
        get_train_loader = get_dali_train_loader(dali_cpu=False)
        train_loader, train_loader_len = get_train_loader(
                                             args.data,
                                             224,#image_size,
                                             args.batch_size,
                                             args.num_classes,
                                             args.mixup > 0.0,
                                             interpolation=args.interpolation,
                                             start_epoch=start_epoch,
                                             workers=args.workers,
                                         )
    else:
        print("Bad databackend picked")
        exit(1)

    if args.data_backend == "pytorch":
        val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                    ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
        val_loader_len = len(val_loader)
    else:
        from image_classification.dataloaders_dali import get_dali_val_loader
        get_val_loader = get_dali_val_loader()
        val_loader, val_loader_len = get_val_loader(
            args.data,
            224, #image_size,
            args.batch_size,
            args.num_classes,
            False,
            interpolation=args.interpolation,
            workers=args.workers,
            )

    if args.mixup != 0.0:
        train_loader = MixUpWrapper(args.mixup, train_loader)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger = log.Logger(
            args.print_freq,
            [
                dllogger.StdOutBackend(
                    dllogger.Verbosity.DEFAULT, step_format=log.format_step
                ),
                dllogger.JSONStreamBackend(
                    dllogger.Verbosity.VERBOSE,
                    os.path.join(args.workspace, args.raport_file),
                ),
            ],
            start_epoch=start_epoch - 1,
        )

    else:
        logger = log.Logger(args.print_freq, [], start_epoch=start_epoch - 1)

    logger.log_parameter(args.__dict__, verbosity=dllogger.Verbosity.DEFAULT)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)
        random.seed(args.seed)
    model_and_loss.load_model_state(model_state)
    if args.device == "mlu":
        model_and_loss.model.to('mlu')
        model_and_loss.loss.to('mlu')

    optimizer = get_optimizer(
        list(model_and_loss.model.named_parameters()),
        args.fp16,
        args.lr,
        args.momentum,
        args.weight_decay,
        nesterov=args.nesterov,
        bn_weight_decay=args.bn_weight_decay,
        state=optimizer_state,
        static_loss_scale=args.static_loss_scale,
        dynamic_loss_scale=args.dynamic_loss_scale,
    )
    if args.device == "mlu":
        ct.to(optimizer, torch.device('mlu'))

    if args.lr_schedule == "step":
        lr_policy = lr_step_policy(
            args.lr, [30, 60, 80], 0.1, args.warmup, logger=logger
        )
    elif args.lr_schedule == "cosine":
        lr_policy = lr_cosine_policy(args.lr, args.warmup, args.epochs, logger=logger)
    elif args.lr_schedule == "linear":
        lr_policy = lr_linear_policy(args.lr, args.warmup, args.epochs, logger=logger)

    if args.amp:
        model_and_loss, optimizer = amp.initialize(
            model_and_loss,
            optimizer,
            opt_level="O1",
            loss_scale="dynamic" if args.dynamic_loss_scale else args.static_loss_scale,
        )
    if args.device == "mlu" and args.cnmix:
        model_and_loss.model, optimizer = cnmix.initialize(
            model_and_loss.model,
            optimizer,
            opt_level=args.opt_level)
        if args.resume:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location='cpu')
                if isinstance(checkpoint, dict) and 'cnmix' in checkpoint:
                    cnmix.load_state_dict(checkpoint['cnmix'])
    if args.distributed:
        model_and_loss.distributed(args)

    if args.device == 'mlu' and args.cnmix:
        cnmix.cnmix_set_amp_quantify_params('all', {'batch_size': args.batch_size,'data_num': args.batch_size * len(train_loader)})

    train_loop(
        args,
        model_and_loss,
        optimizer,
        lr_policy,
        train_loader,
        val_loader,
        args.fp16,
        logger,
        should_backup_checkpoint(args),
        use_amp=args.amp,
        batch_size_multiplier=batch_size_multiplier,
        start_epoch=start_epoch,
        end_epoch=(start_epoch + args.run_epochs)
        if args.run_epochs != -1
        else args.epochs,
        iters=args.iters,
        best_prec1=best_prec1,
        prof=args.prof,
        skip_training=args.evaluate,
        skip_validation=args.training_only,
        save_checkpoints=args.save_checkpoints and not args.evaluate,
        checkpoint_dir=args.workspace,
        checkpoint_filename=args.checkpoint_filename,
        scaler=scaler
    )
    exp_duration = time.time() - exp_start_time
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.end()
        if os.getenv('AVG_LOG'):
            metric_collector = MetricCollector(enable_only_avglog=True)
            metric_collector.insert_metrics(net = "ngc_resnet50v1_5",
                                        accuracy = [logger.metrics['val.top1']['meter'].get_run(), logger.metrics['val.top5']['meter'].get_run()])
            metric_collector.dump()

    print("Experiment ended")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    add_parser_arguments(parser)
    args = parser.parse_args()
    cudnn.benchmark = True
    main(args)
