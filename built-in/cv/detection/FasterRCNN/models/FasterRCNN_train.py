import os
import sys
import argparse
import torch
import random
import numpy as np

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

def train(cfg, args,writer):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    if _USE_MLU == False:
        use_mixed_precision = cfg.DTYPE == "float16"
        amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
    elif cfg.USE_CNMIX:
        cnmix.core.cnmix_set_amp_use_online(True)
        model, optimizer = cnmix.initialize(model, optimizer, opt_level=cfg.CNMIX_OPT_LEVEL)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False, find_unused_parameters=True,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, cfg.USE_CNMIX)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=args.distributed,
        start_iter=arguments["iteration"],
    )

    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=args.distributed, is_for_period=True)
    else:
        data_loader_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
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
    )

    return model

def run_test(cfg, model, distributed):
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
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument('--device', type=str, default='MLU',
                    help='set the type of hardware used for training.')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--prefix", type=str)
    parser.add_argument(
        "--seed", default=None, type=int)
    parser.add_argument(
        "--cnmix",
        action="store_true",
        default=False,
        help="use cnmix for mixed precision training"
    )
    parser.add_argument(
        "--opt_level",
        type=str,
        default="O1",
        help="choose level of mixing precision"
    )

    args = parser.parse_args()
    if args.cnmix:
        args.opts.extend(["USE_CNMIX", True, "CNMIX_OPT_LEVEL", args.opt_level])

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
        
    if args.seed is  None:
        args.seed = np.random.randint(1e4)
        
    if args.distributed:
        args.seed = (args.seed + torch.distributed.get_rank()) % 2**32
    print("Using seed = {}".format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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
    save_config(cfg, output_config_path)

    writer = SummaryWriter("runs/{}_{}".format(args.prefix, args.local_rank))
    model = train(cfg, args, writer)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)

if __name__ == "__main__":
    main()
