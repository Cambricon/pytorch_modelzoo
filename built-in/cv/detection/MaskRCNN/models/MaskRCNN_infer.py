import os
import sys
import argparse
import torch

sys.path.append('./models')

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config

from tensorboardX import SummaryWriter

try:
    import torch_mlu.core.mlu_model as ct
    _USE_MLU = True
except ImportError:
    _USE_MLU = False

if _USE_MLU == False:
    # See if we can use apex.DistributedDataParallel instead of the torch default,
    # and enable mixed-precision via apex.amp
    try:
        from apex import amp
    except ImportError:
        raise ImportError('Use APEX for multi-precision via apex.amp')
else:
    try:
        import cnmix
    except ImportError:
        print('Try to import cnmix failed!!!')

def infer(cfg, model, distributed,args):
    if distributed:
        model = model.module
    torch.mlu.empty_cache() if _USE_MLU else torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            cfg=cfg
        )
        synchronize()
        
def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
                        "--ckpt",
                        help="The path to the checkpoint for test, default is the latest checkpoint.",
                        default=None,
    )
    parser.add_argument("opts",help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--seed", default=None, type=int)
    args = parser.parse_args()

    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    num_devs = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_devs > 1

    if args.distributed:
        if _USE_MLU:
            backend = "cncl"
            ct.set_device(args.local_rank)
        else:
            backend = "nccl"
            torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend=backend, init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} DEVs".format(num_devs))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)
    
    ############################################
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    ############################################
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(args.ckpt, use_latest=args.ckpt is None)
    infer(cfg, model, args.distributed,args)

if __name__ == "__main__":
    main()
