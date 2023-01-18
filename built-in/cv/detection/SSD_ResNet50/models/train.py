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

from torch.autograd import Variable
import torch
import time
from timer import AverageMeter, ProgressMeter
from torch.cuda.amp import autocast
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../tools/utils/")
from metric import MetricCollector

try:
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
except ImportError:
    print("import torch_mlu failed!")
    
def train_loop(model, loss_func, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std, scaler=None):
    epoch_iter = 0
    adaptive_cnt = int(os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT')) if (
            os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT') is not None) else 0
    batch_time_benchmark = []
    batch_time = AverageMeter('Time', ':6.3f')
    batch_loss = AverageMeter('Loss', ':6.10f')
    if args.data_backend == "pytorch":
        progress = ProgressMeter(
            len(train_dataloader),
            [batch_time, batch_loss],
            prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    
    metric_collector = MetricCollector(
        enable_only_benchmark=True,
        record_elapsed_time=True,
        record_hardware_time=True if args.device == "MLU" else False)
    metric_collector.place()
    
    if args.data_backend == "pytorch":
        print("pytorch data_backend")
        for nbatch, (img, ids, img_size, bbox, label) in enumerate(train_dataloader):
            if (epoch_iter == args.iterations):
                break

            if args.device == "MLU":
                img = img.to('mlu', non_blocking=True)
                bbox = bbox.to('mlu', non_blocking=True)
                label = label.to('mlu', non_blocking=True)
            elif args.device == "GPU":
                img = img.cuda()
                bbox = bbox.cuda()
                label = label.cuda()

            with autocast(enabled=args.pyamp):
                ploc, plabel = model(img)
                ploc, plabel = ploc.float(), plabel.float()

                trans_bbox = bbox.transpose(1, 2).contiguous()

                if args.device == "MLU":
                    trans_bbox = trans_bbox.to('mlu', non_blocking=True)
                elif args.device == "GPU":
                    trans_bbox = trans_bbox.cuda()

                gloc = Variable(trans_bbox, requires_grad=False)
                glabel = Variable(label, requires_grad=False)

                loss = loss_func(ploc, plabel, gloc, glabel)

            if args.local_rank == 0:
                logger.update_iter(epoch, iteration, loss.item())
                batch_loss.update(loss.item())

            if args.device == "MLU" and args.cnmix:
                import cnmix
                with cnmix.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            elif args.pyamp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if args.warmup is not None:
                warmup(optim, args.warmup, iteration, args.learning_rate)

            if args.pyamp:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()

            optim.zero_grad()

            if args.device == "GPU":
                torch.cuda.synchronize()
            elif args.device == "MLU":
                torch.mlu.synchronize()

            metric_collector.record()
            metric_collector.place()
            if epoch_iter >= adaptive_cnt:
                batch_time_benchmark.append(time.time() - end)
            batch_time.update(time.time() - end)
            end = time.time()

            iteration += 1
            epoch_iter +=1
    else:
        print("not pytorch data_backend")
        for nbatch, data in enumerate(train_dataloader):
            img = data[0][0][0]
            bbox = data[0][1][0]
            label = data[0][2][0]
            if args.device == "MLU":
                label = label.type(torch.mlu.LongTensor)
            elif args.device == "GPU":
                label = label.type(torch.cuda.LongTensor)
            bbox_offsets = data[0][3][0]
            if args.device == "MLU":
                bbox_offsets = bbox_offsets.mlu()
            elif args.device == "GPU":
                bbox_offsets = bbox_offsets.cuda()
            img.sub_(mean).div_(std)
            if args.device == "GPU":
                img = img.cuda()
                bbox = bbox.cuda()
                label = label.cuda()
                bbox_offsets = bbox_offsets.cuda()
            elif args.device == "MLU":
                img = img.mlu()
                bbox = bbox.mlu()
                label = label.mlu()
                bbox_offsets = bbox_offsets.mlu()

            N = img.shape[0]
            if bbox_offsets[-1].item() == 0:
                print("No labels in batch")
                continue

            # output is ([N*8732, 4], [N*8732], need [N, 8732, 4], [N, 8732] respectively
            M = bbox.shape[0] // N
            bbox = bbox.view(N, M, 4)
            label = label.view(N, M)

            with autocast(enabled=args.pyamp):
                ploc, plabel = model(img)

                ploc, plabel = ploc.float(), plabel.float()
                if args.device == "GPU":
                    trans_bbox = bbox.transpose(1, 2).contiguous().cuda()
                elif args.device == "MLU":
                    trans_bbox = bbox.transpose(1, 2).contiguous().mlu()
                gloc = Variable(trans_bbox, requires_grad=False)
                glabel = Variable(label, requires_grad=False)

                loss = loss_func(ploc, plabel, gloc, glabel)

            if args.warmup is not None:
                warmup(optim, args.warmup, iteration, args.learning_rate)

            if args.pyamp:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
            optim.zero_grad()

            if args.device == "GPU":
                torch.cuda.synchronize()
            elif args.device == "MLU":
                torch.mlu.synchronize()

            metric_collector.record()
            metric_collector.place()
            if epoch_iter >= adaptive_cnt:
                batch_time_benchmark.append(time.time() - end)
            batch_time.update(time.time() - end)
            end = time.time()

            if args.local_rank == 0:
                logger.update_iter(epoch, iteration, loss.item())
            iteration += 1
            epoch_iter += 1

        return iteration
    
    if args.device == "MLU":
        cards = ct.device_count() if args.local_rank == 0 else 1
    if args.device == "GPU":
        cards = torch.cuda.device_count() if args.local_rank == 0 else 1
        
    metrics = metric_collector.get_metrics()
    if 'batch_time_avg' in metrics:
        metric_collector.insert_metrics(
            throughput = args.batch_size  / metrics['batch_time_avg'] * cards)
        
    metric_collector.insert_metrics(
        net = "SSD_ResNet50",
        batch_size = args.batch_size,
        precision = 'amp'if args.pyamp else args.opt_level if args.cnmix else "fp32",
        cards = torch.distributed.get_world_size() if args.distributed else 1,
        DPF_mode = "ddp " if args.distributed == True else "single")
    if (args.distributed == False  or (args.local_rank == 0)):
        metric_collector.dump()

    return iteration

def benchmark_train_loop(model, loss_func, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    hw_time = AverageMeter('HWTime', ':6.3f')
    progress = ProgressMeter(
        len(train_dataloader),
        [batch_time, hw_time, data_time],
        prefix="Epoch: [{}]".format(epoch))


    start_time = None
    # tensor for results
    result = torch.zeros((1,)).to('mlu')
    end = time.time()
    if args.device == "MLU":
        import torch_mlu
        import torch_mlu.core.mlu_model as ct
    for i, (img, _, img_size, bbox, label) in enumerate(train_dataloader):
        data_time.update(time.time() - end)
        if i >= args.benchmark_warmup:
            if args.device == "MLU":
                ct.current_queue().synchronize()
            elif args.device == "GPU":
                torch.cuda.synchronize()
            start_time = time.time()

        if args.device == "MLU":
            import torch_mlu.core.device.notifier as Notifier
            start_t = Notifier.Notifier()
            end_t= Notifier.Notifier()
            start_t.place()

            img = img.to('mlu', non_blocking=True)
            bbox = bbox.to('mlu', non_blocking=True)
            label = label.to('mlu', non_blocking=True)
        elif args.device == "GPU":
            img = img.cuda()
            bbox = bbox.cuda()
            label = label.cuda()

        ploc, plabel = model(img)
        ploc, plabel = ploc.float(), plabel.float()

        trans_bbox = bbox.transpose(1, 2).contiguous()

        if args.device == "MLU":
            trans_bbox = trans_bbox.to('mlu', non_blocking=True)
        elif args.device == "GPU":
            trans_bbox.cuda()
        gloc = Variable(trans_bbox, requires_grad=False)
        glabel = Variable(label, requires_grad=False)

        loss = loss_func(ploc, plabel, gloc, glabel)

        if args.device == "MLU" and args.cnmix:
            import cnmix
            with cnmix.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optim.step()
        optim.zero_grad()

        if args.device == "MLU":
            end_t.place()

        if i >= args.benchmark_warmup + args.benchmark_iterations:
            break

        if i >= args.benchmark_warmup:
            if args.device == "MLU":
                ct.current_queue().synchronize()
            elif args.device == "GPU":
                torch.cuda.synchronize()
            logger.update(args.batch_size, time.time() - start_time)

        if args.device == "MLU":
            end_t.synchronize()
            hwtime = start_t.elapsed_time(end_t)
            hw_time.update(hwtime / 1000./ 1000.)
        batch_time.update(time.time() - end)
        end = time.time()
        iteration += 1
        progress.display(iteration)



    result.data[0] = logger.print_result()
    if args.N_gpu > 1:
        torch.distributed.reduce(result, 0)
    if args.local_rank == 0:
        print('Training performance = {} FPS'.format(float(result.data[0])))
    if os.getenv('AVG_LOG'):
        with open(os.getenv('AVG_LOG'), 'a') as train_out:
            train_out.write('net:SSD_Resnet50, FPS:{}, '.format(float(result.data[0])))


def loop(dataloader, reset=True):
    while True:
        for data in dataloader:
            yield data
        if reset:
            dataloader.reset()

def benchmark_inference_loop(model, loss_func, epoch, optim, train_dataloader, val_dataloader, encoder, iteration, logger, args, mean, std):
    assert args.N_gpu == 1, 'Inference benchmark only on 1 gpu'
    start_time = None
    model.eval()

    i = -1
    val_datas = loop(val_dataloader, False)

    while True:
        i += 1
        torch.cuda.synchronize()
        if i >= args.benchmark_warmup:
            start_time = time.time()

        data = next(val_datas)

        with torch.no_grad():
            img = data[0]
            if not args.no_cuda:
                img = img.cuda()
            img.sub_(mean).div_(std)
            img = Variable(img, requires_grad=False)
            _ = model(img)
            torch.cuda.synchronize()

            if i >= args.benchmark_warmup + args.benchmark_iterations:
                break

            if i >= args.benchmark_warmup:
                logger.update(args.eval_batch_size, time.time() - start_time)

    logger.print_result()

def warmup(optim, warmup_iters, iteration, base_lr):
    if iteration < warmup_iters:
        new_lr = 1. * base_lr / warmup_iters * iteration
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr


def load_checkpoint(model, checkpoint):
    """
    Load model from checkpoint.
    """
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint, map_location=torch.device('cpu'))

    # remove proceeding 'N.' from checkpoint that comes from DDP wrapper
    saved_model = od["model"]
    model.load_state_dict(saved_model, strict=False)


def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay}]
