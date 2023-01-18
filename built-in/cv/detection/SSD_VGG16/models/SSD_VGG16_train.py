import sys
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../tools/utils/")
sys.path.append(cur_dir + "/models")
from metric import MetricCollector
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd

import re
import random
import sys
import time
import warnings

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--pretrained_path', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--device', default='mlu', type=str,
                    help='Use MLU or GPU to train model, defaultly be MLU')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--iters', type=int, default=1000, metavar='N',
                    help='train how many iterations totaly')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://127.0.0.10:28500', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--device_id', default=None, type=int,
                    help='Use specified device for training, useless in '
                         'multiprocessing distributed training')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--pyamp', action='store_true', default=False,
                    help='use pytorch amp for mixed precision training')
parser.add_argument('--cnmix', action='store_true', help='use cnmix')
parser.add_argument('--opt_level', type=str, default='O0', help='cnmix optimizer level')
args = parser.parse_args()

if args.device == 'mlu':
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
    print("Use MLU Deivce ......")
    if args.cnmix:
        try:
            import cnmix
        except ImportError:
            print("MLU Training without cnmix!")

if torch.cuda.is_available():
    if args.device == 'gpu':
        # There are no such APIs like torch.mlu.FloatTensor, in order to
        # maintains the consistensy before we support torch.mlu.FloatTensor,
        # we temporarily disable the torch.cuda.FloatTensor.
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.set_default_tensor_type('torch.FloatTensor')
    if args.device != 'gpu':
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def load_amp_scaler_state(scaler, base_file):
    other, ext = os.path.splitext(base_file)
    if ext == '.pkl' or '.pth':
        ckpt = torch.load(base_file, map_location=lambda storage, loc: storage)
        if isinstance(ckpt, dict) and 'amp' in ckpt:
            print('Loading the scaler state of AMP...')
            scaler.load_state_dict(ckpt['amp'])
            print('Finished!')

def load_cnmix_state_dict(cnmix, base_file):
    other, ext = os.path.splitext(base_file)
    if ext == '.pkl' or '.pth':
        ckpt = torch.load(base_file, map_location=lambda storage, loc: storage)
        if isinstance(ckpt, dict) and 'cnmix' in ckpt:
            print('Loading the scaler state of CNMIX...')
            cnmix.load_state_dict(ckpt['cnmix'])
            print('Finished!')
def main():
    print('Training SSD on:', args.dataset)
    print('Using the specified args:')
    print(args)
    f = open("param_train.txt", "w")
    f.write(str(args)+"\n")
    f.close()

    if args.device_id is not None:
        warnings.warn('You have chosen a specific device. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.multiprocessing_distributed or args.world_size > 1

    ndevs_per_node = ct.device_count() if args.device == 'mlu' else torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ndevs_per_node * args.world_size
        mp.spawn(train, nprocs=ndevs_per_node, args=(ndevs_per_node, args))
    else:
        train(args.device_id, ndevs_per_node, args)

def train(dev_id, ndevs_per_node, args):
    batch_time = AverageMeter('Time' , ':6.3f')
    losses     = AverageMeter('Loss' , ':.4e' )
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            if os.getenv('BENCHMARK_LOG') is None:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.benchmark = True

    args.device_id = dev_id
    if args.device_id is not None:
        print("Use Device: {} for training".format(args.device_id))

    if args.device == 'mlu':
        ct.set_device(args.device_id)
    elif args.device == 'gpu':
        torch.cuda.set_device(args.device_id)
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ndevs_per_node + dev_id
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)

    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
    if args.distributed:
        args.batch_size = int(args.batch_size / ndevs_per_node)
        args.num_workers = int((args.num_workers + ndevs_per_node - 1) / ndevs_per_node)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None

    if args.visdom:
        import visdom
        viz = visdom.Visdom()
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.pyamp:
        scaler = GradScaler()

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
        if args.pyamp:
            load_amp_scaler_state(scaler, args.resume)
        if args.cnmix:
            load_cnmix_state_dict(cnmix, args.resume)
    else:
        vgg_weights = torch.load(args.pretrained_path + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    if args.device == 'gpu':
        net = net.cuda()
        net.priors = net.priors.cuda()
    elif args.device == 'mlu':
        iters_per_epoch = len(dataset) // args.batch_size
        if train_sampler is not None:
            iters_per_epoch = iters_per_epoch // ndevs_per_node
        print("iters_per_epoch is: ", iters_per_epoch)
        net.to(ct.mlu_device())
        net.priors = net.priors.to('mlu')
    else:
        print('using CPU, this will be slow')

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  sampler=train_sampler,
                                  shuffle=(train_sampler is None),
                                  collate_fn=detection_collate,
                                  pin_memory=True)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.device)
    if args.device == 'mlu':
        ct.to(optimizer, torch.device('mlu'))

    if args.device == 'mlu' and args.cnmix:
        net, optimizer = cnmix.initialize(net, optimizer, opt_level=args.opt_level)
        cnmix.cnmix_set_amp_quantify_params(
                'all', {'batch_size': args.batch_size, 'data_num': args.batch_size * len(data_loader)})

    if args.device != 'cpu' and args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.device_id])
    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')
    epoch_size = len(dataset) // args.batch_size
    step_index = 0
    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    #writer = SummaryWriter('runs/fp')
    # create batch iterator
    #batch_iterator = iter(data_loader)
    iteration = args.start_iter
    cfg['max_iter'] = iteration + args.iters

    # for internal benchmark test
    metric_collector = MetricCollector(
        enable_only_benchmark=True,
        record_elapsed_time=True,
        record_hardware_time=True if args.device == 'mlu' else False)
    metric_collector.place()
    save_frq = cfg['max_iter']
    while iteration <= cfg['max_iter']:
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
        if args.distributed:
            train_sampler.set_epoch(epoch)
        t1 = time.time()
        for _, (images, targets) in enumerate(data_loader):
            iteration += 1
            if iteration in cfg['lr_steps']:
                step_index += 1
            adjust_learning_rate(optimizer, args.gamma, iteration, step_index)

            if args.device == 'gpu':
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]
            elif args.device == 'mlu':
                images = images.to("mlu", non_blocking=True)
                targets = [ann.to("mlu", non_blocking=True) for ann in targets]
            else:
                images = images
                targets = [ann for ann in targets]
            # forward
            t0 = time.time()

            with autocast(enabled=args.pyamp):
                out = net(images)
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c

            optimizer.zero_grad()
            metric_collector.record()
            metric_collector.place()
            losses.update(loss.item(), images.size(0))
            # backprop
            if args.pyamp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif args.cnmix:
                with cnmix.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                optimizer.step()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            batch_time.update(time.time() - t1)
            t1 = time.time()

            #writer.add_scalar('train/loss', loss.item(), iteration)
            if iteration % 1 == 0:
                print('proc id: ' + str(args.device_id) + ' || iter ' +
                      repr(iteration) + ' || Loss: %.4f ||' % (loss.item()) +
                      ' timer: %.4f sec.' % (t1 - t0))

            if iteration % save_frq == 0:
                if (args.distributed == False) or (args.rank == 0):
                    if args.distributed:
                        state_dict = net.module.state_dict()
                    else:
                        state_dict = net.state_dict()
                    if args.pyamp:
                        state_dict = {"state_dict":state_dict, "amp":scaler.state_dict()}
                    elif args.cnmix:
                        state_dict = {"state_dict":state_dict, "cnmix":cnmix.state_dict()}
                    if args.device == 'mlu':
                        save_pre = "mlu_weights"
                    elif args.device == 'gpu':
                        save_pre = "gpu_weights"
                    else:
                        save_pre = "cpu_weights"
                    print('Saving state, iter:', iteration)
                    if not os.path.exists(args.save_folder):
                        os.makedirs(args.save_folder)
                    torch.save(state_dict, os.path.join(args.save_folder,save_pre + '_ssd300_VOC_' +
                        repr(iteration) + '.pth'))
                break
        epoch += 1
    if ((args.distributed == False) or (args.rank == 0)) and os.getenv('AVG_LOG'):
        with open(os.getenv('AVG_LOG'), 'a') as train_avg:
            train_avg.write('net:SSD-VGG16, iter:{}, cards:{}, avg_loss:{}, avg_time:{}, '.
            format(-1 if args.iters == 120000 else args.iters, args.world_size if args.distributed else 1,
            losses.avg, batch_time.avg))
    if args.cnmix:
        precision = args.opt_level
    elif args.pyamp:
        precision = "amp"
    else:
        precision = "fp32"
        
    # if args.device == "MLU":
    #     cards = ct.device_count() if args.local_rank == 0 else 1
    # if args.device == "GPU":
    #     cards = torch.cuda.device_count() if args.local_rank == 0 else 1   
    metric_collector.insert_metrics(
        net = "SSD_VGG16",
        batch_size = args.batch_size,
        precision = precision,
        cards = args.world_size if args.distributed else 1,
        DPF_mode = "ddp" if args.distributed else "single")
    if ((args.distributed == False) or (args.rank == 0)):
        metric_collector.dump()

def adjust_learning_rate(optimizer, gamma, iteration, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if iteration < 1000:
        lr = args.lr * (iteration // 258 + 1) / (1000 // 258 + 1)
    else:
        lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

if __name__ == '__main__':
    main()
