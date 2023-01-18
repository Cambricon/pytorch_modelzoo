# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize, is_main_process
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.inference import inference

import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../../../tools/utils/")
from metric import MetricCollector

try:
    import torch_mlu.core.mlu_model as ct
except ImportError:
    print("Try to import torch_mlu failed!!!")
    from apex import amp

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    writer,
    args
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    

    scaler = None
        
    start_training_time = time.time()
    end = time.time()

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST

    # for internal benchmark test
    metric_collector = MetricCollector(
        enable_only_benchmark=True,
        record_elapsed_time=True,
        record_hardware_time=True if device.type == 'mlu' else False)
    metric_collector.place()

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):

        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
            
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        metric_collector.record()
        metric_collector.place()
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        
        optimizer.zero_grad()
        if device.type == 'mlu' and cfg.USE_CNMIX:
            import cnmix
            with cnmix.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()
        else:
            losses.backward()
            
        optimizer.step()
            
        scheduler.step()
        
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=(torch.mlu.max_memory_allocated() if device.type == 'mlu' else
                            torch.cuda.max_memory_allocated()) / 1024.0 / 1024.0,
                )
            )
        writer.add_scalar("train/loss", meters.loss.global_avg, iteration)
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], iteration)
        if iteration % checkpoint_period == 0:
            if device.type == 'mlu' and cfg.USE_CNMIX:
                arguments.update({"cnmix": cnmix.state_dict()})
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            
        if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
            meters_val = MetricLogger(delimiter="  ")
            synchronize()
            _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                model,
                # The method changes the segmentation mask format in a data loader,
                # so every time a new data loader is created:
                make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_for_period=True),
                dataset_name="[Validation]",
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=None,
            )
            synchronize()
            model.train()
            with torch.no_grad():
                # Should be one image for each GPU:
                for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                    images_val = images_val.to(device)
                    targets_val = [target.to(device) for target in targets_val]
                    loss_dict = model(images_val, targets_val)
                    losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters_val.update(loss=losses_reduced, **loss_dict_reduced)
            synchronize()
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=(torch.mlu.max_memory_allocated() if device.type == 'mlu' else
                            torch.cuda.max_memory_allocated()) / 1024.0 / 1024.0,
                )
            )
            writer.add_scalar('val/loss', meters_val.loss.global_avg, iteration)
        if max_iter - start_iter > 1000 and (iteration + 1) % int(max_iter // 10) == 0:  # MLU have precheckin(2 iters) and daily(1000 iters) test and do not need save ckpt here
            if device.type == 'mlu' and cfg.USE_CNMIX:
                arguments.update({"cnmix": cnmix.state_dict()})
            checkpointer.save("model_{}".format(iteration), **arguments)
        if iteration == max_iter:
            if device.type == 'mlu' and cfg.USE_CNMIX:
                arguments.update({"cnmix": cnmix.state_dict()})
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    if os.getenv('AVG_LOG') and is_main_process():
        with open(os.getenv('AVG_LOG'), 'a') as train_avg:
            network_name = "MaskRCNN" if "maskrcnn" in cfg.OUTPUT_DIR else "FasterRCNN"
            train_avg.write('net:{}-ResNet101+FPN, iter:{}, cards:{}, avg_loss:{}, avg_time:{}, '.format(network_name, meters.time.count, get_world_size(), meters.loss.global_avg, meters.time.global_avg))
    # insert metrics and dump metrics
    dev_cnt = get_world_size() 
    metric_collector.insert_metrics(
        net = "{}-ResNet101+FPN".format("MaskRCNN" if "maskrcnn" in cfg.OUTPUT_DIR else "FasterRCNN"),
        batch_size = int(cfg.SOLVER.IMS_PER_BATCH / dev_cnt),
        precision = cfg.CNMIX_OPT_LEVEL if cfg.USE_CNMIX else "fp32",
        cards = dev_cnt,
        DPF_mode = "ddp " if dev_cnt > 1 else "single")
    if is_main_process():
        metric_collector.dump()
