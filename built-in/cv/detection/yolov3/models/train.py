import test  # import test.py to get mAP after each epoch
import re
from train_models import *
from utils.datasets import *
from utils.utils_train import *

import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../tools/utils/")
from metric import MetricCollector

parser = argparse.ArgumentParser(description='Pytorch YoloV3 Training.')
# data parameters
parser.add_argument(
    '--data', type=str, default='data/coco2014.data', help='*.data path')
parser.add_argument(
    '--img-size', nargs='+', type=int, default=[416], help='train and test image-sizes')
parser.add_argument(
    '--cache-images', action='store_true', help='cache images for faster training')
# model paramters
parser.add_argument(
    '--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path')
parser.add_argument(
    '--arc', type=str, default='default', help='yolo architecture')  # default, uCE, uBCE
# train paramters
parser.add_argument(
    '--adam', action='store_true', help='use adam optimizer')
parser.add_argument(
    '--benchmark', action='store_true', help='benchmark')
parser.add_argument(
    '--epochs', type=int, default=273, help='total training epochs')
parser.add_argument(
    '--iters', type=int, default=-1, help='total training iters for one epoch')
parser.add_argument(
    '--eval-iters', type=int, default=-1, help='total eval iters for one epoch')
parser.add_argument(
    '--batch-size', type=int, default=16)  # batch_size = 64
parser.add_argument(
    '--multi-scale', action='store_true', help='adjust img_size every 10 batches')
parser.add_argument(
    '--rect', action='store_true', help='rectangular training')
parser.add_argument(
    '--resume', action='store_true', help='resume training from last.pt')
parser.add_argument(
    '--nosave', action='store_true', help='only save final checkpoint')
parser.add_argument(
    '--notest', action='store_true', help='no test')
parser.add_argument(
    '--weights', type=str, default='yolov3/darknet53.conv.74',
    help='initial weights')
parser.add_argument(
    '--bucket', type=str, default='', help='gsutil bucket')
# distirbute learning parameters
parser.add_argument(
    '--distributed', action='store_true', help='use multi-mlu training.')
parser.add_argument(
    '--device', default='mlu', type=str, help='Use cpu gpu or mlu device')
parser.add_argument(
    '--world-size', default=1, type=int,
    help='number of nodes for distributed training')
parser.add_argument(
    '--workers', default=4, type=int, metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--rank', default=0, type=int,
    help='node rank for distributed training')
parser.add_argument(
    '--dist-url', default='tcp://127.0.0.5:29400', type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='nccl', type=str,
    help='distributed backend')
parser.add_argument(
    '--seed', default=66, type=int,
    help='seed for initializing training. ')
parser.add_argument(
    '--ckp-path',type=str,default='./ckps',
    help='Where to save ckps')
parser.add_argument(
    '--logdir', type=str, default='./logs', metavar='PATH',
    help='Where to save logs')

# for cnmix
parser.add_argument('--cnmix', action='store_true',
                    help='use cnmix.')
parser.add_argument('--opt_level', type=str, default='O3',
                    help='the level of cnmix.')

# for pytorch amp
parser.add_argument('--pyamp', action='store_true', default=False,
                    help='use pytorch amp for mixed precision training')

# Hyperparameters (results68: 59.9 mAP@0.5 yolov3-spp-416) https://github.com/ultralytics/yolov3/issues/310

hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 49.5,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.00579,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.5,  # focal loss gamma
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98,  # image rotation (+/- deg)
       'translate': 0.05,  # image translation (+/- fraction)
       'scale': 0.05,  # image scale (+/- gain)
       'shear': 0.641}  # image shear (+/- deg)

args = parser.parse_args()

if args.device == 'mlu':
    import torch_mlu
    import torch_mlu.core.mlu_model as ct

def main():
    print("ARGUMENTS: ", args)
    ndevs_per_node = 1
    if args.device == 'mlu':
        if args.distributed:
            ndevs_per_node = ct.device_count()
        else:
            ndevs_per_node = 1
    elif torch.cuda.is_available() or args.device == 'gpu':
        if args.distributed:
            ndevs_per_node = torch.cuda.device_count()
        else:
            ndevs_per_node = 1
    else:
        args.distributed = False
        print("MLU and GPU is not available, use CPU for training.")

    if args.dist_url == 'env://' and args.world_size == -1:
        args.world_size = int(os.environ['WORLD_SIZE'])
    if args.distributed:
        args.world_size = ndevs_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ndevs_per_node, args=(ndevs_per_node, args))
    else:
        main_worker(args.rank, ndevs_per_node, args)


def main_worker(dev_id, ndevs_per_node, args):
    args.rank = dev_id
    if args.rank is not None:
        if args.device == 'mlu':
            print("Use MLU: {} for training".format(args.rank))
        elif args.device == 'gpu':
            print("Use GPU: {} for training".format(args.rank))
        else:
            print("Use CPU for training")

    if args.seed is not None:
        np.random.seed(args.seed + dev_id)
        random.seed(args.seed + dev_id)
        torch.manual_seed(args.seed + dev_id)
        torch.backends.cudnn.deterministic = True

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.device == 'mlu':
            ct.set_device(args.rank)
        elif args.device == 'gpu':
            torch.cuda.set_device(args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)

    cfg = args.cfg
    data = args.data
    if len(args.img_size) >= 2:
        train_img_size, test_img_size = args.img_size
    else:
        train_img_size = test_img_size = args.img_size[0]
    epochs = args.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    if args.distributed:
        # args.batch_size is total batch size, while batch_size is part of total batch size
        batch_size = int(args.batch_size / ndevs_per_node)
    else:
        batch_size = int(args.batch_size)
    weights = args.weights  # initial training weights

    if args.multi_scale:
        img_sz_min = round(train_img_size / 32 / 1.5)
        img_sz_max = round(train_img_size / 32 * 1.5)
        train_img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('Using multi-scale %g - %g' % (img_sz_min * 32, train_img_size))

    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = int(data_dict['classes'])  # number of classes

    # Initialize model
    model = Darknet(cfg, arc=args.arc)
    # Train set and loader
    train_set = LoadImagesAndLabels(train_path,
                                    train_img_size,
                                    batch_size,
                                    augment=True,
                                    hyp=hyp,  # augmentation hyperparameters
                                    rect=args.rect,  # rectangular training
                                    cache_labels=False,
                                    cache_images=False,
                                    single_cls=False)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               num_workers=args.workers,
                                               shuffle=not args.distributed,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               collate_fn=train_set.collate_fn)

    # Test set and loader
    test_set = LoadImagesAndLabels(test_path,
                                   test_img_size,
                                   batch_size*2,
                                   hyp=hyp,
                                   rect=True,
                                   cache_labels=False,
                                   cache_images=False,
                                   single_cls=False)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size*2,
                                              num_workers=4,
                                              shuffle=False,
                                              pin_memory=True,
                                              collate_fn=test_set.collate_fn)
    if args.device == 'mlu':
        model.to(ct.mlu_device())
    elif args.device == 'gpu':
        model = model.to('cuda:' + str(args.rank))
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else
    if args.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    start_epoch = 0
    best_fitness = 0.0
    if weights.endswith('.pt') or weights.endswith('.pth') or weights.endswith('.pth.tar'):  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        if args.device != 'gpu':
            ckpt = torch.load(weights, map_location='cpu')
        else:
            ckpt = torch.load(weights)

        # load model from *.pt
        try:
            ckpt['model'] = {k: v for k, v in ckpt['model'].items()
                    if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (args.weights, args.cfg, args.weights)
            raise KeyError(s) from e

        if args.resume:
            start_epoch = ckpt['epoch']
            best_fitness = ckpt['best_fitness']
            optimizer.load_state_dict(ckpt['optimizer'])

    elif len(weights) > 0:  # darknet format
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        load_darknet_weights(model, weights)

    # Start training
    nb = len(train_loader)
    prebias = start_epoch == 0
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    if not args.distributed or (args.rank == 0):
        print('Using %g dataloader workers' % args.workers)
        print('Starting training for %g epochs...' % epochs)

    if args.device == "mlu" and args.cnmix:
        import cnmix
        model, optimizer = cnmix.initialize(model, optimizer, opt_level=args.opt_level)
        cnmix.core.cnmix_set_amp_quantify_params(
            'all', {'batch_size': args.batch_size, 'data_num': args.batch_size * nb})
        if args.resume and 'cnmix' in ckpt:
            cnmix.load_state_dict(ckpt['cnmix'])
    
    scaler = None
    if args.pyamp:
        scaler = GradScaler()
        if args.resume and 'amp' in ckpt:
            scaler.load_state_dict(ckpt['amp'])

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.rank], find_unused_parameters=True, broadcast_buffers=True)

    model.nc = nc  # attach number of classes to model
    model.arc = args.arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(args.epochs * x) for x in \
                    [0.8, 0.9]], gamma=0.1, last_epoch=start_epoch - 1)

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------
        model.train()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # Prebias
        if prebias:
            if epoch < 3:  # prebias
                ps = 0.1, 0.9  # prebias settings (lr=0.1, momentum=0.9)
            else:  # normal training
                ps = hyp['lr0'], hyp['momentum']  # normal training settings
                prebias = False

            # Bias optimizer settings
            optimizer.param_groups[2]['lr'] = ps[0]
            if optimizer.param_groups[2].get('momentum') is not None:  # for SGD but not Adam
                optimizer.param_groups[2]['momentum'] = ps[1]

        mloss = torch.zeros(4)  # mean losses
        if args.device == 'mlu':
            mloss = mloss.to(ct.mlu_device())
        if args.device == 'gpu':
            mloss = mloss.cuda(args.rank)

        nb = len(train_loader)
        args.iters = len(train_loader) if len(train_loader) < args.iters or args.iters == -1 else args.iters

        # for internal benchmark test
        metric_collector = MetricCollector(
            enable_only_benchmark=True,
            record_elapsed_time=True,
            record_hardware_time=True if args.device == 'mlu' else False)
        metric_collector.place()

        for i, (imgs, targets, paths, _) in enumerate(train_loader): #pbar:  # batch -------------------------------------------------------------
            if i == args.iters:
                break
            ni = i + nb * epoch # number integrated batches (since train start)
            imgs = Variable(imgs, requires_grad=False)
            targets = Variable(targets, requires_grad=False)
            if args.device == 'mlu':
                imgs = imgs.to(ct.mlu_device(), non_blocking=True).float() / 255.0
                targets = targets.to(ct.mlu_device(), non_blocking=True)
            elif args.device == 'gpu':
                imgs = imgs.cuda(args.rank, non_blocking=True).float() / 255.0
                targets = targets.cuda(args.rank, non_blocking=True)
            else:
                imgs = imgs.float() / 255.0

            if args.multi_scale:
                train_img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = train_img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    # new shape (stretched to 32-multiple)
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]
                    imgs = F.interpolate(
                        imgs, size=ns, mode='bilinear', align_corners=False)

            with autocast(enabled=args.pyamp):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets, model, not prebias)

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results
            optimizer.zero_grad()

            if args.device == "mlu" and args.cnmix:
                import cnmix
                with cnmix.scale_loss(loss,optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif args.pyamp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if args.pyamp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            if not args.distributed or (args.rank % ndevs_per_node == 0):
                print("Epoch {}/{}, Iter {}/{}, IoU {:.3f}, Obj {:.3f}, Class {:.3f}, Total {:.3f}".format(
                      epoch, epochs, i, args.iters, mloss[0].item(), mloss[1].item(),
                      mloss[2].item(), mloss[3].item()))

            # MetricCollector record
            metric_collector.record()
            metric_collector.place()

        final_epoch = epoch + 1 == epochs
        if not args.notest:  # or final_epoch:  # Calculate mAP
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
            results, maps = test.test(task="test",
                                      cfg=args.cfg,
                                      data=args.data,
                                      batch_size=batch_size * 2,
                                      eval_iters=args.eval_iters,
                                      img_size=test_img_size,
                                      model=model,
                                      conf_thres=1E-3 if(final_epoch and is_coco) else 0.1,  # 0.1 faster
                                      iou_thres=0.63,
                                      save_json=False,
                                      single_cls=False,
                                      dataloader=test_loader)

        # Update scheduler
        scheduler.step()

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            is_best = True
            best_fitness = fi
        else:
            is_best = False

        if final_epoch or (not args.nosave):
            # Create checkpoint
            ckpt = {
                'epoch': epoch + 1,
                'best_fitness': best_fitness,
                'optimizer': optimizer.state_dict()
            }

            if args.device == "mlu" and args.cnmix:
                import cnmix
                ckpt['cnmix'] = cnmix.state_dict()

            if args.pyamp and scaler is not None:
                ckpt['amp'] = scaler.state_dict()


            if not os.path.exists(args.ckp_path) and args.rank == 0:
                os.makedirs(args.ckp_path)

            multi_mlu = type(model) in (nn.parallel.DataParallel,
                                        nn.parallel.DistributedDataParallel)

            if ((epoch + 1) % 10 == 0) or final_epoch:
                ckpt['model'] = model.module.state_dict() if multi_mlu else model.state_dict()
                if not args.distributed or (args.rank % ndevs_per_node == 0):
                    print("save checkpoint :", 'epoch_{}.pth'.format(epoch))
                    save_checkpoint(ckpt, dir_path=args.ckp_path, filename='epoch_{}.pth'.format(epoch))

            if is_best:
                ckpt['model'] = model.module.state_dict() if multi_mlu else model.state_dict()
                if not args.distributed or (args.rank % ndevs_per_node == 0):
                    print("save checkpoint :", 'model_best.pth.tar')
                    save_checkpoint(ckpt, dir_path=args.ckp_path, filename = 'model_best.pth.tar')
            del ckpt

        if not os.path.exists(args.logdir + '_epoch') and args.rank == 0:
            os.makedirs(args.logdir + '_epoch')
        s = "Epoch {}, valid/P {:.3f}, valid/R {:.3f}, valid/mAP {:.3f}, valid/f1 {:.3f}\n".format(
            epoch + 1, results[0], results[1], results[2], results[3])
        if not args.notest and args.rank == 0:
            print(s)

        if is_best and args.rank == 0:
            s = "Best mAP: Epoch {}, valid/P {:.3f}, valid/R {:.3f}, valid/mAP {:.3f}, valid/f1 {:.3f}\n".format(
                epoch + 1, results[0], results[1], results[2], results[3])

            if not os.path.exists(args.logdir + '_epoch') and args.rank == 0:
                os.makedirs(args.logdir + '_epoch')
            if not args.distributed or (args.rank % ndevs_per_node == 0):
                train_f = open(args.logdir + '_epoch' + '/ddp_0_' + str(epoch) +
                               '_' + str(args.batch_size) + '.csv', 'a+')
                train_f.write(s)
                train_f.close()

        if args.cnmix:
            precision = args.opt_level
        elif args.pyamp:
            precision = "amp"
        else:
            precision = "fp32"
        # insert metrics and dump metrics
        metric_collector.insert_metrics(
            net = "yolov3",
            batch_size = batch_size,
            precision = precision,
            cards = args.world_size if args.distributed else 1,
            DPF_mode = "ddp " if args.distributed == True else "single")
        if not args.distributed or (args.rank % ndevs_per_node == 0):
            metric_collector.dump()

        # end epoch ----------------------------------------------------------------------------------------------------

    # destroy the process group
    if args.distributed:
        dist.destroy_process_group()
    elif args.device == 'gpu':
        torch.cuda.empty_cache()
    return results


def save_checkpoint(state,dir_path,filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(dir_path, filename))

if __name__ == '__main__':
    main()
