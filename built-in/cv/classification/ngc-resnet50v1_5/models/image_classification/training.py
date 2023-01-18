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
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
from . import logger as log
from . import resnet as models
from . import utils

import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../tools/utils/")
from metric import MetricCollector

try:
    import torch_mlu
    from torch_mlu.core.device.notifier import Notifier
    import torch_mlu.core.mlu_model as ct
    from torch.nn.parallel import DistributedDataParallel as DDP_mlu
except ImportError:
    print("Import torch_mlu failed!")

try:
    import cnmix
except ImportError:
    print("Import cnmix failed!")

import dllogger
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    print(
        "Import apex error! Please install apex from https://www.github.com/nvidia/apex to run this example."
    )


ACC_METADATA = {"unit": "%", "format": ":.2f"}
IPS_METADATA = {"unit": "img/s", "format": ":.2f"}
TIME_METADATA = {"unit": "s", "format": ":.5f"}
LOSS_METADATA = {"format": ":.5f"}

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

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

class ModelAndLoss(nn.Module):
    def __init__(
        self,
        arch,
        loss,
        pretrained_weights=None,
        device="mlu",
        fp16=False,
        memory_format=torch.contiguous_format,
    ):
        super(ModelAndLoss, self).__init__()
        self.arch = arch

        print("=> creating model '{}'".format(arch))
        model = models.build_resnet(arch[0], arch[1], arch[2])
        if pretrained_weights is not None:
            print("=> using pre-trained model from a file '{}'".format(arch))
            model.load_state_dict(pretrained_weights)

        global device_global
        device_global = device
        if device == "gpu":
            model.cuda()
        if fp16:
            model = network_to_half(model)

        # define loss function (criterion) and optimizer
        criterion = loss()

        if device == "gpu":
            criterion = criterion.cuda()

        self.model = model
        self.loss = criterion

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss(output, target)

        return loss, output

    def distributed(self, args):
        if device_global == "mlu":
            self.model = DDP_mlu(self.model, device_ids=[args.mlu])
        else:
            self.model = DDP(self.model)

    def load_model_state(self, state):
        if not state is None:
            self.model.load_state_dict(state, strict=False)


def get_optimizer(
    parameters,
    fp16,
    lr,
    momentum,
    weight_decay,
    nesterov=False,
    state=None,
    static_loss_scale=1.0,
    dynamic_loss_scale=False,
    bn_weight_decay=False,
):

    if bn_weight_decay:
        print(" ! Weight decay applied to BN parameters ")
        optimizer = torch.optim.SGD(
            [v for n, v in parameters],
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    else:
        print(" ! Weight decay NOT applied to BN parameters ")
        bn_params = [v for n, v in parameters if "bn" in n]
        rest_params = [v for n, v in parameters if not "bn" in n]
        print(len(bn_params))
        print(len(rest_params))
        optimizer = torch.optim.SGD(
            [
                {"params": bn_params, "weight_decay": 0},
                {"params": rest_params, "weight_decay": weight_decay},
            ],
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    if fp16:
        optimizer = FP16_Optimizer(
            optimizer,
            static_loss_scale=static_loss_scale,
            dynamic_loss_scale=dynamic_loss_scale,
            verbose=False,
        )

    if not state is None:
        optimizer.load_state_dict(state)

    return optimizer


def lr_policy(lr_fn, logger=None):
    if logger is not None:
        logger.register_metric(
            "lr", log.LR_METER(), verbosity=dllogger.Verbosity.VERBOSE
        )

    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        if logger is not None:
            logger.log_metric("lr", lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_linear_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1 - (e / es))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_cosine_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_exponential_policy(
    base_lr, warmup_length, epochs, final_multiplier=0.001, logger=None
):
    es = epochs - warmup_length
    epoch_decay = np.power(2, np.log2(final_multiplier) / es)

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * (epoch_decay ** e)
        return lr

    return lr_policy(_lr_fn, logger=logger)


def get_train_step(
    model_and_loss, optimizer, args, fp16, use_amp=False, batch_size_multiplier=1, scaler=None
):
    def _step(input, target, optimizer_step=True, args=args, scaler=scaler):
        input_var = Variable(input)
        target_var = Variable(target)
        with autocast(enabled=args.pyamp):
            loss, output = model_and_loss(input_var, target_var)
            if torch.distributed.is_initialized():
                reduced_loss = utils.reduce_tensor(loss.data, args.device)
            else:
                reduced_loss = loss.data

        if fp16:
            optimizer.backward(loss)
        elif use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        elif args.device == "mlu" and args.cnmix:
            with cnmix.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        elif args.pyamp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if optimizer_step:
            opt = optimizer
            for param_group in opt.param_groups:
                for param in param_group["params"]:
                    param.grad /= batch_size_multiplier

            if args.pyamp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        if args.device == "mlu":
            ct.current_queue().synchronize()
        else:
            torch.cuda.synchronize()

        return reduced_loss

    return _step


def train(
    args,
    train_loader,
    model_and_loss,
    optimizer,
    lr_scheduler,
    fp16,
    logger,
    epoch,
    iters,
    use_amp=False,
    prof=-1,
    batch_size_multiplier=1,
    register_metrics=True,
    scaler=None
):

    if register_metrics and logger is not None:
        logger.register_metric(
            "train.loss",
            log.LOSS_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=LOSS_METADATA,
        )
        logger.register_metric(
            "train.compute_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "train.total_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "train.data_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "train.compute_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )

    # for internal benchmark test
    metric_collector = MetricCollector(
        enable_only_benchmark=True,
        record_elapsed_time=True,
        record_hardware_time=True if args.device == 'mlu' else False)
    metric_collector.place()

    step = get_train_step(
        model_and_loss,
        optimizer,
        args,
        fp16,
        use_amp=use_amp,
        batch_size_multiplier=batch_size_multiplier,
        scaler=scaler
    )

    model_and_loss.train()
    end = time.time()

    optimizer.zero_grad()

    data_iter = enumerate(train_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter)
    if prof > 0:
        data_iter = utils.first_n(prof, data_iter)
    if args.dummy_test:
        data_iter = enumerate(dummy_data_loader(len = len(train_loader), batch_size = args.batch_size))
    adaptive_cnt = int(os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT')) if (
            os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT') is not None) else 0
    for i, (input, target) in data_iter:
        if i == iters:
            break
        bs = input.size(0)
        lr_scheduler(optimizer, i, epoch)
        data_time = time.time() - end

        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        if not args.dummy_test:
            if args.device == "mlu":
                input = input.to('mlu', non_blocking=True)
                target = target.to('mlu', non_blocking=True)
            elif args.device == "gpu":
                input = input.cuda()
                target = target.cuda()
        loss = step(input, target, optimizer_step=optimizer_step, args=args)

        it_time = time.time() - end

        if logger is not None:
            # In order to print the information of the first 100 iters, record all.
            # When step = 100, clear the current log to achieve the original i >= adaptive_ ant.
            logger.log_metric("train.loss", to_python_float(loss), bs)
            logger.log_metric("train.total_time", it_time)
            logger.log_metric("train.data_time", data_time)
            logger.log_metric("train.compute_time", it_time - data_time)
            if i == adaptive_cnt:
                logger.reset('train.loss')
                logger.reset('train.total_time')
                logger.reset('train.data_time')
                logger.reset('train.compute_time')

        # MetricCollector record
        metric_collector.record()
        metric_collector.place()

        end = time.time()

    dev_cnt = 1 if not torch.distributed.is_initialized() else int(os.environ["WORLD_SIZE"])
    precision = 'fp32'
    if args.cnmix:
        precision = args.opt_level
    if args.pyamp:
        precision = 'amp'
    metric_collector.insert_metrics(
        net = "ngc_resnet50v1_5",
        batch_size = args.batch_size,
        precision = precision,
        cards = dev_cnt,
        DPF_mode = "ddp" if torch.distributed.is_initialized() else "single")
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        metric_collector.dump()


def get_val_step(model_and_loss, args):
    def _step(input, target, args=args):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():
            loss, output = model_and_loss(input_var, target_var)

        prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized():
            reduced_loss = utils.reduce_tensor(loss.data, args.device)
            prec1 = utils.reduce_tensor(prec1, args.device)
            prec5 = utils.reduce_tensor(prec5, args.device)
        else:
            reduced_loss = loss.data

        if args.device == "mlu":
            ct.current_queue().synchronize()
        else:
            torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def validate(
    args, val_loader, model_and_loss, fp16, logger, epoch, prof=-1, register_metrics=True
):
    if register_metrics and logger is not None:
        logger.register_metric(
            "val.top1",
            log.ACC_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=ACC_METADATA,
        )
        logger.register_metric(
            "val.top5",
            log.ACC_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=ACC_METADATA,
        )
        logger.register_metric(
            "val.loss",
            log.LOSS_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=LOSS_METADATA,
        )
        logger.register_metric(
            "val.compute_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "val.total_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "val.data_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at100",
            log.LAT_100(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at99",
            log.LAT_99(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at95",
            log.LAT_95(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )

    step = get_val_step(model_and_loss, args)

    top1 = log.AverageMeter()
    # switch to evaluate mode
    model_and_loss.eval()

    end = time.time()

    data_iter = enumerate(val_loader)
    if not logger is None:
        data_iter = logger.iteration_generator_wrapper(data_iter, val=True)
    if prof > 0:
        data_iter = utils.first_n(prof, data_iter)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        data_time = time.time() - end

        if args.device == "mlu":
            loss, prec1, prec5 = step(input.to('mlu'), target.to('mlu'), args=args)
        elif args.device == "gpu":
            loss, prec1, prec5 = step(input.cuda(), target.cuda(), args=args)
        else:
            loss, prec1, prec5 = step(input, target, args=args)

        it_time = time.time() - end

        top1.record(to_python_float(prec1), bs)
        if logger is not None:
            logger.log_metric("val.top1", to_python_float(prec1), bs)
            logger.log_metric("val.top5", to_python_float(prec5), bs)
            logger.log_metric("val.loss", to_python_float(loss), bs)
            logger.log_metric("val.compute_ips", calc_ips(bs, it_time - data_time))
            logger.log_metric("val.total_ips", calc_ips(bs, it_time))
            logger.log_metric("val.data_time", data_time)
            logger.log_metric("val.compute_latency", it_time - data_time)
            logger.log_metric("val.compute_latency_at95", it_time - data_time)
            logger.log_metric("val.compute_latency_at99", it_time - data_time)
            logger.log_metric("val.compute_latency_at100", it_time - data_time)

        end = time.time()

    return top1.get_val()


# Train loop {{{
def calc_ips(batch_size, time):
    world_size = (
            torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    tbs = world_size * batch_size
    return tbs / time

def train_loop(
    args,
    model_and_loss,
    optimizer,
    lr_scheduler,
    train_loader,
    val_loader,
    fp16,
    logger,
    should_backup_checkpoint,
    use_amp=False,
    batch_size_multiplier=1,
    best_prec1=0,
    start_epoch=0,
    end_epoch=0,
    iters=-1,
    prof=-1,
    skip_training=False,
    skip_validation=False,
    save_checkpoints=True,
    checkpoint_dir="./",
    checkpoint_filename="checkpoint.pth.tar",
    scaler=None
):

    prec1 = -1

    print("RUNNING EPOCHS FROM {} TO {}".format(start_epoch,end_epoch))
    for epoch in range(start_epoch, end_epoch):
        if logger is not None:
            logger.start_epoch()
        if not skip_training:
            train(
                args,
                train_loader,
                model_and_loss,
                optimizer,
                lr_scheduler,
                fp16,
                logger,
                epoch,
                iters,
                use_amp=use_amp,
                prof=prof,
                register_metrics=epoch == start_epoch,
                batch_size_multiplier=batch_size_multiplier,
                scaler=scaler
            )

        if not skip_validation:
            prec1, nimg = validate(
                args,
                val_loader,
                model_and_loss,
                fp16,
                logger,
                epoch,
                prof=prof,
                register_metrics=epoch == start_epoch,
            )
        if logger is not None:
            logger.end_epoch()

        if save_checkpoints and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            if not skip_validation:
                is_best = logger.metrics["val.top1"]["meter"].get_epoch() > best_prec1
                best_prec1 = max(
                    logger.metrics["val.top1"]["meter"].get_epoch(), best_prec1
                )
            else:
                is_best = False
                best_prec1 = 0

            if should_backup_checkpoint(epoch):
                backup_filename = "checkpoint-{}.pth.tar".format(epoch + 1)
            else:
                backup_filename = None
            save_params={
                    "epoch": epoch + 1,
                    "arch": model_and_loss.arch,
                    "state_dict": model_and_loss.model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                }
            if args.device == "mlu" and args.cnmix:
                save_params["cnmix"] = cnmix.state_dict()
            if args.pyamp:
                save_params["amp"] = scaler.state_dict()

            utils.save_checkpoint(
                args,
                save_params,
                is_best,
                checkpoint_dir=checkpoint_dir,
                backup_filename=backup_filename,
                filename=checkpoint_filename,
            )


# }}}
