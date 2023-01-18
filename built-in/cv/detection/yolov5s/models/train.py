import argparse

import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.datasets import *
from utils.utils import *

import sys
import numpy as np
from copy import deepcopy
import re

import torch.multiprocessing as mp
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../tools/utils/")
from metric import MetricCollector

# Close shared memory
default_collate_func = dataloader.default_collate

def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]

# Hyperparameters
hyp = {'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 5e-4,  # optimizer weight decay
       'giou': 0.05,  # giou loss gain
       'cls': 0.58,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'anchor_t': 4.0,  # anchor-multiple threshold
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.014,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.68,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 0.0,  # image rotation (+/- deg)
       'translate': 0.0,  # image translation (+/- fraction)
       'scale': 0.5,  # image scale (+/- gain)
       'shear': 0.0}  # image shear (+/- deg)
print(hyp)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

# Print focal loss if gamma > 0
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='*.cfg path')
parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
parser.add_argument('--rect', action='store_true', help='rectangular training')
parser.add_argument('--resume', action='store_true', help='resume training')
parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
parser.add_argument('--notest', action='store_true', help='only test final epoch')
parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')

parser.add_argument('--weights', type=str, default='', help='initial weights path')
parser.add_argument('--weights-ema', type=str, default='', help='ema weights path')

parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
parser.add_argument('--device', default='mlu', type=str, help='Use cpu or mlu or gpu device')
parser.add_argument('--adam', action='store_true', help='use adam optimizer')
parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%')
parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')

# MLU DDP
parser.add_argument('--multiprocessing-distributed', action='store_true', help='use multiprocessing training.')
parser.add_argument('--device-id', default=0, type=int, help='device id')
parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--workers', default=16, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:8812', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='cncl', type=str, help='distributed backend')
parser.add_argument('--seed', default=66, type=int, help='seed for initializing training. ')
parser.add_argument('--ckp-path',type=str, default='./weights/', help='Where to save ckps')
parser.add_argument('--iters', type=int, default=-1, help='total training iters for one epoch')
parser.add_argument('--eval-iters', type=int, default=-1, help='size of test iters')

# Cnmix
parser.add_argument('--cnmix', action='store_true', help='use cnmix.')
parser.add_argument('--opt_level', type=str, default='O0', help='the level of cnmix.')

# for pytorch amp
parser.add_argument('--pyamp', action='store_true', default=False,
                    help='use pytorch amp for mixed precision training')

opt = parser.parse_args()
opt.cfg = glob.glob('./**/' + opt.cfg, recursive=True)[0]  # find file
opt.data = glob.glob('./**/' + opt.data, recursive=True)[0]  # find file
print(opt)
opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)

results_file = 'results_{}.txt'.format(opt.device)

mixed_precision = False
if opt.device == 'mlu':
    import torch_mlu.core.mlu_model as ct
    if opt.cnmix:
        try:
            import cnmix
            mixed_precision = True
        except ImportError:
            print("MLU Training without cnmix!")
if opt.device == 'gpu':
    try:  # Mixed precision training https://github.com/NVIDIA/apex
        from apex import amp
        mixed_precision = True
    except:
        print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')

def main_worker(dev, ndevs_per_node=None, opt=None):
    if opt.multiprocessing_distributed:
        writer = SummaryWriter('runs_{}_ddp/{}_{}_{}'.format(opt.device, opt.device, dev, ndevs_per_node))
    else:
        writer = SummaryWriter('runs_{}_single/{}_{}'.format(opt.device, opt.device, dev))

    opt.device_id = dev
    if opt.device_id is not None:
        print("Use {}: {} for training".format(opt.device, opt.device_id))

    if opt.seed is not None:
        random.seed(opt.seed + dev)
        np.random.seed(opt.seed + dev)
        torch.manual_seed(opt.seed + dev)
        torch.cuda.manual_seed(opt.seed + dev)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if opt.multiprocessing_distributed:
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])
        opt.rank = opt.rank * ndevs_per_node + dev
        opt.workers = int((opt.workers + ndevs_per_node - 1) / ndevs_per_node)
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

    epochs = opt.epochs  # 300
    weights = opt.weights  # initial training weights

    # Configure
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    train_path = data_dict['train']
    test_path = data_dict['val']

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes

    '''# Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)'''

    # Create model
    model = Model(opt.cfg)
    assert model.md['nc'] == nc, '%s nc=%g classes but %s nc=%g classes' % (opt.data, nc, opt.cfg, model.md['nc'])

    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride)
    if any(x % gs != 0 for x in opt.img_size):
        print('WARNING: --img-size %g,%g must be multiple of %s max stride %g' % (*opt.img_size, opt.cfg, gs))
    imgsz, imgsz_test = [make_divisible(x, gs) for x in opt.img_size]  # image sizes (train, test)

    if opt.multiprocessing_distributed and os.getenv('BENCHMARK_LOG') is None:
        batch_size = int(opt.batch_size / ndevs_per_node)
    else:
        batch_size = int(opt.batch_size)


    # Trainset
    train_set = LoadImagesAndLabels(
        train_path,
        imgsz,
        batch_size,
        augment=True,
        hyp=hyp,  # augmentation hyperparameters
        rect=opt.rect,  # rectangular training
        cache_images=opt.cache_images,
        single_cls=opt.single_cls)
    mlc = np.concatenate(train_set.labels, 0)[:, 0].max()  # max label class
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Correct your labels or your model.' % (mlc, nc, opt.cfg)

    if opt.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    # Trainloader
    batch_size = min(batch_size, len(train_set))
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=opt.workers,
        shuffle=not opt.rect and not opt.multiprocessing_distributed,  # Shuffle=True unless rectangular training is used
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=train_set.collate_fn)

    # Testset
    test_set = LoadImagesAndLabels(
        test_path,
        imgsz_test,
        batch_size,
        hyp=hyp,
        rect=True,
        cache_images=opt.cache_images,
        single_cls=opt.single_cls)

    # Testloader
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=opt.workers,
        pin_memory=True,
        collate_fn=test_set.collate_fn)

    if opt.device == 'mlu':
        ct.set_device(opt.device_id)
        model.to(ct.mlu_device())

    if opt.device == 'gpu':
        torch.cuda.set_device(opt.device_id)
        model.cuda(opt.device_id)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                pg2.append(v)  # biases
            elif '.weight' in k and '.bn' not in k:
                pg1.append(v)  # apply weight decay
            else:
                pg0.append(v)  # all else

    optimizer = optim.Adam(pg0, lr=hyp['lr0']) if opt.adam else \
        optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Load Model
    # For resume training, load both origin model and ema model, use origin model to train.
    start_epoch, best_fitness = 0, 0.0
    if weights.endswith('.pt') or weights.endswith('.pth') or weights.endswith('.pth.tar'):  # pytorch format
        if opt.device != 'gpu':
            ckpt = torch.load(weights, map_location='cpu')  # load checkpoint
        else:
            ckpt = torch.load(weights)
        # load model
        try:
            # ckpt['model'] = \
            #    {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s." \
                % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        if opt.resume:
            start_epoch = ckpt['epoch']
            best_fitness = ckpt['best_fitness']
            optimizer.load_state_dict(ckpt['optimizer'])

        start_epoch = ckpt['epoch'] + 1

    if opt.device == 'mlu':
        if mixed_precision:
            model, optimizer = cnmix.initialize(model, optimizer, opt_level=opt.opt_level)
            cnmix.cnmix_set_amp_quantify_params(
                'all', {'batch_size': batch_size, 'data_num': batch_size * len(train_loader)})
            if opt.resume and ckpt.get('cnmix') is not None:
                cnmix.load_state_dict(ckpt['cnmix'])

    scaler = None
    if opt.pyamp:
        scaler = GradScaler()
        if opt.resume and 'amp' in ckpt:
            scaler.load_state_dict(ckpt['amp'])


    if opt.device == 'gpu':
        # Mixed precision training https://github.com/NVIDIA/apex
        if mixed_precision:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1  # do not move
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    if opt.multiprocessing_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[opt.device_id], broadcast_buffers=True)

    # Model parameters
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(train_set.labels, nc) #.to(device)  # attach class weights
    model.names = data_dict['names']

    # class frequency
    labels = np.concatenate(train_set.labels, 0)
    c = torch.tensor(labels[:, 0])  # classes
    # cf = torch.bincount(c.long(), minlength=nc) + 1.
    # model._initialize_biases(cf.to(device))
    # plot_labels(labels)  #<----------------------------close by xujing
    writer.add_histogram('classes', c, 0)

    # Exponential moving average
    ema = torch_utils.ModelEMA(model)

    # Load ema model
    if opt.resume:
        if opt.device != 'gpu':
            ckpt = torch.load(opt.weights_ema, map_location='cpu')
        else:
            ckpt = torch.load(opt.weights_ema)

        if type(ema.ema) is nn.parallel.DistributedDataParallel:
            ema.ema.module.load_state_dict(ckpt['model'], strict=False)
        else:
            ema.ema.load_state_dict(ckpt['model'], strict=False)

        ema.updates = ckpt['updates']
        del ckpt

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    n_burn = max(3 * nb, 1e3)  # burn-in iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
    print('Using %g dataloader workers' % opt.workers)
    print('Starting training for %g epochs...' % epochs)
    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        if opt.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)

        model.train()

        # Update image weights (optional)
        if train_set.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(train_set.labels, nc=nc, class_weights=w)
            train_set.indices = random.choices(range(train_set.n), weights=image_weights, k=train_set.n)  # rand weighted idx

        mloss = torch.zeros(4, dtype=torch.float)  # mean losses
        if opt.device == 'gpu':
            mloss = mloss.cuda(opt.device_id)
        if opt.device == 'mlu':
            mloss = mloss.to(ct.mlu_device(), non_blocking=True)

        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(train_loader, total=nb)  # progress bar

        # for internal benchmark test
        metric_collector = MetricCollector(
            enable_only_benchmark=True,
            record_elapsed_time=True,
            record_hardware_time=True if opt.device == 'mlu' else False)
        metric_collector.place()

        for i, (imgs, targets, paths, _) in enumerate(pbar):  # batch -------------------------------------------------------------
            if i == opt.iters:
                break

            ni = i + nb * epoch # number integrated batches (since train start)

            if opt.device == 'mlu':
                imgs = imgs.to(ct.mlu_device(), non_blocking=True)
                imgs = imgs.float() / 255.0 # uint8 to float32, 0 - 255 to 0.0 - 1.0
                targets = targets.to(ct.mlu_device(), non_blocking=True)
            elif opt.device == 'gpu':
                imgs = imgs.cuda(opt.device_id)
                targets = targets.cuda(opt.device_id)
                imgs = imgs.float() / 255.0
            else:
                imgs = imgs.float() / 255.0

            # Burn-in
            if ni <= n_burn:
                xi = [0, n_burn]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            with autocast(enabled=opt.pyamp):
                # Forward
                pred = model(imgs)
                # Loss
                loss, loss_items = compute_loss(pred, targets, model)

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Backward
            if opt.device == 'mlu' and mixed_precision:
                with cnmix.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif opt.device == 'gpu' and mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif opt.pyamp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimize
            if ni % accumulate == 0:
                if opt.pyamp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)

            # Plot
            # if ni < 3:
            #    f = 'train_batch%g.jpg' % i  # filename
            #    res = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
            #    if writer:
            #        writer.add_image(f, res, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # MetricCollector record
            metric_collector.record()
            metric_collector.place()

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        # mAP
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs

        # Use model to validate not ema model
        if not opt.notest: # or final_epoch:  # Calculate mAP
            results, maps, times = test.test(opt.data,
                                            batch_size=batch_size,
                                            imgsz=imgsz_test,
                                            save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                            model=model,
                                            single_cls=opt.single_cls,
                                            dataloader=test_loader,
                                            fast=ni < n_burn,
                                            eval_iters=opt.eval_iters)

        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(opt.name) and opt.bucket:
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

        # Tensorboard
        if writer:
            tags = ['train/giou_loss',
                    'train/obj_loss',
                    'train/cls_loss',
                    'metrics/precision',
                    'metrics/recall',
                    'metrics/mAP_0.5',
                    'metrics/F1',
                    'val/giou_loss',
                    'val/obj_loss',
                    'val/cls_loss']
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                writer.add_scalar(tag, x, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            is_best = True
            best_fitness = fi
        else:
            is_best = False

        # Save model
        save = (not opt.nosave) or (final_epoch)
        if save:
            if not opt.multiprocessing_distributed or (opt.rank % ndevs_per_node == 0): #save:
                with open(results_file, 'r') as f:  # create checkpoint
                    model_state = model.module.state_dict() if type(model) is nn.parallel.DistributedDataParallel else model.state_dict()
                    ema_state = ema.ema.module.state_dict() if type(ema.ema) is nn.parallel.DistributedDataParallel else ema.ema.state_dict()

                    # ema ckpt
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema_state,
                            'optimizer': None if final_epoch else optimizer.state_dict(),
                            'updates': ema.updates}
                    # origin ckpt
                    ckpt_origin = {'epoch': epoch,
                                   'best_fitness': best_fitness,
                                   'training_results': f.read(),
                                   'model': model_state,
                                   'optimizer': None if final_epoch else optimizer.state_dict()}

                    # qparam of ema model is same with origin model
                    if opt.device == 'mlu' and mixed_precision:
                        ckpt['cnmix'] = cnmix.state_dict()
                        ckpt_origin['cnmix'] = cnmix.state_dict()

                    if opt.pyamp and scaler is not None:
                        ckpt['amp'] = scaler.state_dict()
                        ckpt_origin['amp'] = scaler.state_dict()

                    dir_path = os.path.join(opt.ckp_path, opt.device)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)

                    save_checkpoint(ckpt, False, dir_path=dir_path, filename='epoch_{}.pth'.format(epoch))
                    save_checkpoint(ckpt_origin, False, dir_path=dir_path, filename='origin_epoch_{}.pth'.format(epoch))
                    save_checkpoint(ckpt, is_best, dir_path=dir_path)

                    del ckpt

        if opt.cnmix:
            precision = opt.opt_level
        elif opt.pyamp:
            precision = "amp"
        else:
            precision = "fp32"

        # insert metrics and dump metrics
        metric_collector.insert_metrics(
            net = "yolov5s",
            batch_size = opt.batch_size,
            precision = precision,
            cards = opt.world_size if opt.multiprocessing_distributed else 1,
            DPF_mode = "ddp " if opt.multiprocessing_distributed == True else "single")
        if not opt.multiprocessing_distributed or (opt.rank % ndevs_per_node == 0):
            metric_collector.dump()

        # end epoch ----------------------------------------------------------------------------------------------------

    # end training
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    if opt.multiprocessing_distributed:
        dist.destroy_process_group()
    if opt.device == 'gpu':
        torch.cuda.empty_cache()
    return results

def save_checkpoint(state, is_best, dir_path, filename="checkpoint.pth.tar"):
    if is_best:
        filename = "model_best.pth.tar"
    torch.save(state, os.path.join(dir_path, filename))

if __name__ == '__main__':
    # Train
    if opt.dist_url == 'env://' and opt.world_size == -1:
        opt.world_size = int(os.environ['WORLD_SIZE'])

    ndevs_per_node = 1
    if opt.device == 'mlu':
        if opt.multiprocessing_distributed:
            ndevs_per_node = ct.device_count()
        else:
            ndevs_per_node = 1
    elif torch.cuda.is_available() and opt.device == 'gpu':
        if opt.multiprocessing_distributed:
            ndevs_per_node = torch.cuda.device_count()
        else:
            ndevs_per_node = 1
    else:
        opt.multiprocessing_distributed = False

    if opt.multiprocessing_distributed:
        opt.world_size = ndevs_per_node * opt.world_size
        mp.spawn(main_worker, nprocs=ndevs_per_node, args=(ndevs_per_node, opt))
    else:
        main_worker(opt.device_id, ndevs_per_node, opt)
