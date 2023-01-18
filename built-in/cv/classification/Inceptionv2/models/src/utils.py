import random
import math
import os
import shutil


import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.utils.data
import numpy as np
import yaml


from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
import models
from collections import OrderedDict

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

optimizer_names = ['sgd']
criterion_names = ['crossentropy']
scheduler_names = ['step', 'cos_simple']
input_calib_types = ['max', 'max_channel', 'pytorch', 'histeqnm']
weight_calib_types = ['max']
train_modes = ['auto', 'fix', 'fix_all', 'auto_bit', 'fix_bit', 'fix_all_bit']
infer_modes = ['infer_online', 'infer_offline']

class ToSpaceBGR(object):

    def __init__(self, is_bgr=True):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255=True):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor

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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



class Scheduler(object):
    def __init__(self, config, init_lr, epoch_iter):
        self.init_lr = init_lr
        self.config = config
        self.epoch_iter = epoch_iter
        if self.config['type'] == 'cos_simple':
            self.warmup_iter = self.config['warmup_epoch'] * self.epoch_iter
            self.max_iter = self.config['epochs'] * self.epoch_iter
        self.cur_lr = init_lr

    def get_lr(self, epoch, iters):
        if self.config['type'] == 'step':
            lr = self.init_lr * (self.config['gamma'] ** (epoch // self.config['step']))
        elif self.config['type'] == 'cos_simple':
            if epoch < self.config['warmup_epoch']:
                lr = self.init_lr * iters / self.warmup_iter
            else:
                lr = self.init_lr *( 1 + math.cos(math.pi*(iters - self.warmup_iter)/(self.max_iter - self.warmup_iter))) /2
        else:
            raise ValueError
        self.cur_lr = lr
        return lr


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calcu_bit(model, mode='all'):
    if type(model) == torch.nn.parallel.DistributedDataParallel:
        state = model.module.state_dict()
    else:
        state = model.state_dict()
    bit_num = {8:0, 16:0, 31:0}
    all_num = 0
    for key in state.keys():
        if 'bit' in key:
            bit_num[int(state[key].item())] += 1
            all_num += 1
    for key in bit_num.keys():
        bit_num[key] = float(bit_num[key])/float(all_num)

    return bit_num



def calcu_class_acc(output, target, topk=(1,)):
    r"""Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output (torch.tensor, dim is [N, Class_Num]): output of the model
        target (torch.tensor, dim is [N]): the label of class
        topk (tuple, default=(1,)): the topk index

    Return:
        res (tuple): the topk acc

    """
    with torch.no_grad():
        maxk, batch_size = max(topk), target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def init_dist(config, rank, total, device='cuda'):
    r"""Initial the distribute env use args.
    """
    config['rank'] = config['rank'] * total + rank
    dist.init_process_group(
        backend=config['dist_backend'], init_method=config['dist_url'],
        world_size=config['world_size'], rank=config['rank'])

    if os.getenv('BENCHMARK_LOG') is None:
        config['batch_size'] = int(config['batch_size'] / total)
        config['workers']= int((config['workers'] + total - 1) / total)
    if config['batch_size'] % total != 0:
        print("Please check the batch size {} can be divided by gpus {}".format(
            config['batch_size'], total))
        raise ValueError


def init_seed(seed=None, use_gpu=False):
    r"""Initial the seed of the torch, random, numpy

    Args:
        seed (int, default=None): seed of the program, set if the seed is not None
        use_gpu (bool, default=False): set gpu seed

    """
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_gpu:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.benchmark = False

        print('Set seed to {} in random, numpy and torch,'
              ' please check if you use other packages that need seed.'.format(seed))


def convert_ddp_model(model, rank=None, device='cuda'):
    r"""Convert model to ddp model

    Args:
        model (torch.nn.module): torch.nn.module
        rank (int, default=None): the rank of the ddp model

    Returns:
        model (torch.nn

    """
    if rank is not None:
        model = parallel.DistributedDataParallel(
            model, device_ids=[rank], broadcast_buffers=False)
    else:
        model = parallel.DistributedDataParallel(
            model, broadcast_buffers=False)

    return model


def create_criterion(config):
    if config['type'] == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError

    return criterion


def create_dataset(data_config, transform):
    dataset = datasets.ImageFolder(data_config['path'], transform)
    return dataset


def create_loader(data_config, loader_config, distributed=False, ndevs_per_node=1, dev=1, device='cuda'):
    transform = create_transform(data_config['transform'])
    dataset = create_dataset(data_config, transform)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=loader_config['batch_size'],
        shuffle=loader_config['shuffle'] and sampler is None,
        num_workers=loader_config['workers'],
        pin_memory=loader_config['pin_memory'],
        sampler=sampler)

    return loader, sampler

def remove_module_prefix(state):
    state_keys = list(state.keys())
    for key in state_keys:
        if key.startswith('module.'):
             state[key[7:]] = state[key]
             state.pop(key)


def create_model(arch, pretrained=False, pretrained_ckp=None, args=None):
    r"""Load the model use arch and pretrained model

    Args:
        arch (str): model name
        pretrained (bool, default=False): use the pretrained model or not
        pretrained_ckp (str, default=None): pretrained path
    """
    model = models.__dict__[arch]()
    if pretrained:
        if pretrained_ckp:
            print("=> using pre-trained model {} for model {} ".format(pretrained_ckp, arch))
            state = torch.load(pretrained_ckp, map_location='cpu')#['state_dict']
            if args.pyamp:
                if isinstance(state, dict) and 'amp' in state:
                    args.scaler.load_state_dict(state['amp'])
            #new_sd = OrderedDict()
            #for k, v in state.items():
            #    new_sd[k[7:]] = v

            #model.load_state_dict(new_sd, strict=False)
            if 'model' in state.keys():
                state = state['model']
            remove_module_prefix(state)

            if 'densenet' in arch:
                pattern = re.compile(
                    r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
                for key in list(state.keys()):
                    res = pattern.match(key)
                    if res:
                        new_key = res.group(1) + res.group(2)
                        state[new_key] = state[key]
                        del state[key]

            state_keys = list(state.keys())
            model_keys = list(model.state_dict().keys())
            miss_keys = []
            for key in model_keys:
                if key not in state_keys:
                    miss_keys.append(key)
            if len(miss_keys) > 1:
                print('Please check the pre-trained model, the keys {} is missing.'.format(miss_keys))
            model.load_state_dict(state, strict=False)
        else:
            print("Please provide the right pre-trained model, not {}.".format(pretrained_ckp))
            raise ValueError
    return model


def create_optimizer(model, config):
    if config['type'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'])
    else:
        raise ValueError

    return optimizer

'''
def create_transform(trans_config):
    trans_list = []
    for trans in trans_config['list']:
        if trans == 'resize':
            trans_list.append(transforms.Resize(trans_config['resize']))
        elif trans == 'randomresizedcrop':
            trans_list.append(transforms.RandomResizedCrop(trans_config['crop']))
        elif trans == 'randomhorizontalflip':
            trans_list.append(transforms.RandomHorizontalFlip())
        elif trans == 'centercrop':
            trans_list.append(transforms.CenterCrop(trans_config['crop']))
        elif trans == 'totensor':
            trans_list.append(transforms.ToTensor())
        elif trans == 'normalize':
            trans_list.append(
                transforms.Normalize(
                    mean=trans_config['mean'], std=trans_config['std']))
        else:
            print('Please check transform type {}.'.format(trans))
            raise ValueError
    transform = transforms.Compose(trans_list)
    return transform
'''

def create_transform(trans_config):
    trans_list = []
    for trans in trans_config['list']:
        if trans == 'resize':
            trans_list.append(transforms.Resize(trans_config['resize']))
        elif trans == 'randomresizedcrop':
            trans_list.append(transforms.RandomResizedCrop(trans_config['crop']))
        elif trans == 'randomhorizontalflip':
            trans_list.append(transforms.RandomHorizontalFlip())
        elif trans == 'centercrop':
            trans_list.append(transforms.CenterCrop(trans_config['crop']))
        elif trans == 'totensor':
            trans_list.append(transforms.ToTensor())
        elif trans == 'tospacebgr':
            trans_list.append(ToSpaceBGR())
        elif trans == 'torange255':
            trans_list.append(ToRange255())
        elif trans == 'normalize':
            trans_list.append(
                transforms.Normalize(
                    mean=trans_config['mean'], std=trans_config['std']))
        else:
            print('Please check transform type {}.'.format(trans))
            raise ValueError
    transform = transforms.Compose(trans_list)
    return transform


def precess_train_config(args):
    if not os.path.exists(args.config):
        print('Please check the config {}.'.format(args.config))
        raise ValueError
    with open(args.config, 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.arch:
        config['model'] = args.arch
    if args.pretrained and args.pretrained_ckp:
        config['pretrain']['path'] = args.pretrained_ckp
    if args.train_dataset:
        config['train_dataset']['path'] = args.train_dataset
    if args.train_batch_size:
        config['train']['batch_size'] = args.train_batch_size
    if args.learning_rate:
        config['train']['optimizer']['learning_rate'] = args.learning_rate
    if args.optimizer:
        config['train']['optimizer']['type'] = args.optimizer
    if args.momentum:
        config['train']['optimizer']['momentum'] = args.momentum
    if args.weight_decay:
        config['train']['optimizer']['weight_decay'] = args.weight_decay
    if args.scheduler:
        config['train']['scheduler']['type'] = args.scheduler
    if args.scheduler_step:
        config['train']['scheduler']['step'] = args.scheduler_step
    if args.epochs:
        config['train']['scheduler']['epochs'] = args.epochs
    if args.scheduler_gamma:
        config['train']['scheduler']['gamma'] = args.scheduler_gamma
    if args.warmup_epoch:
        config['train']['scheduler']['warmup_epoch'] = args.warmup_epoch
    if args.criterion:
        config['train']['criterion']['type'] = args.cretrion

    if args.world_size:
        config['train']['world_size'] = args.world_size
    if args.rank:
        config['train']['rank'] = args.rank
    if args.dist_url:
        config['train']['dist_url'] = args.dist_url
    if args.dist_backend:
        config['train']['dist_backend'] = args.dist_backend

    if args.valid_dataset:
        config['valid_dataset']['path'] = args.valid_dataset
    if args.valid_batch_size:
        config['valid']['batch_size'] = args.valid_batch_size
    if args.valid_workers:
        config['valid']['workers'] = args.valid_workers

    return config


def precess_train_fix_config(args):
    if not os.path.exists(args.config):
        print('Please check the config {}.'.format(args.config))
        raise ValueError
    with open(args.config, 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.arch:
        config['model'] = args.arch
    if args.pretrained and args.pretrained_ckp:
        config['pretrain']['path'] = args.pretrained_ckp
    if args.train_dataset:
        config['train_dataset']['path'] = args.train_dataset
    if args.train_batch_size:
        config['train']['batch_size'] = args.train_batch_size
    if args.learning_rate:
        config['train']['optimizer']['learning_rate'] = args.learning_rate
    if args.optimizer:
        config['train']['optimizer']['type'] = args.optimizer
    if args.momentum:
        config['train']['optimizer']['momentum'] = args.momentum
    if args.weight_decay:
        config['train']['optimizer']['weight_decay'] = args.weight_decay
    if args.epochs:
        config['train']['scheduler']['epochs'] = args.epochs
    if args.scheduler:
        config['train']['scheduler']['type'] = args.scheduler
    if args.warmup_epoch:
        config['train']['scheduler']['warmup_epoch'] = args.warmup_epoch
    if args.scheduler_step:
        config['train']['scheduler']['step'] = args.scheduler_step
    if args.scheduler_gamma:
        config['train']['scheduler']['gamma'] = args.scheduler_gamma
    if args.criterion:
        config['train']['criterion']['type'] = args.cretrion

    if args.world_size:
        config['train']['world_size'] = args.world_size
    if args.rank:
        config['train']['rank'] = args.rank
    if args.dist_url:
        config['train']['dist_url'] = args.dist_url
    if args.dist_backend:
        config['train']['dist_backend'] = args.dist_backend

    if args.valid_dataset:
        config['valid_dataset']['path'] = args.valid_dataset
    if args.valid_batch_size:
        config['valid']['batch_size'] = args.valid_batch_size
    if args.valid_workers:
        config['valid']['workers'] = args.valid_workers

    if args.second_stage_iters:
        config['quanz']['second_stage_iters'] = args.second_stage_iters
    if args.max_update_iters:
        config['quanz']['max_update_iters'] = args.max_update_iters
    if args.input_init_bit:
        config['quanz']['input_init_bit'] = args.input_init_bit
    if args.input_mode:
        config['quanz']['input_mode'] = args.input_mode
    if args.weight_init_bit:
        config['quanz']['weight_init_bit'] = args.weight_init_bit
    if args.weight_mode:
        config['quanz']['weight_mode'] = args.weight_mode
    if args.grad_init_bit:
        config['quanz']['grad_init_bit'] = args.grad_init_bit
    if args.grad_mode:
        config['quanz']['grad_mode'] = args.grad_mode
    if args.infer:
        config['quanz']['infer'] = args.infer


    return config


def precess_valid_config(args):
    if not os.path.exists(args.config):
        print('Please check the config {}.'.format(args.config))
        raise ValueError
    with open(args.config, 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.arch:
        config['model'] = args.arch
    if args.pretrained and args.pretrained_ckp:
        config['pretrain']['path'] = args.pretrained_ckp
    if args.valid_dataset:
        config['valid_dataset']['path'] = args.valid_dataset
    if args.valid_batch_size:
        config['valid']['batch_size'] = args.valid_batch_size
    if args.valid_workers:
        config['valid']['workers'] = args.valid_workers

    return config


def precess_valid_fix_config(args):
    if not os.path.exists(args.config):
        print('Please check the config {}.'.format(args.config))
        raise ValueError
    with open(args.config, 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.arch:
        config['model'] = args.arch
    if args.pretrained and args.pretrained_ckp:
        config['pretrain']['path'] = args.pretrained_ckp
    if args.valid_dataset:
        config['valid_dataset']['path'] = args.valid_dataset
    if args.valid_batch_size:
        config['valid']['batch_size'] = args.valid_batch_size
    if args.valid_workers:
        config['valid']['workers'] = args.valid_workers
    if args.calib_dataset:
        config['calib_dataset']['path'] = args.calib_dataset
    if args.calib_file:
        config['calib_dataset']['file'] = args.calib_file
    if args.calib_batch_size:
        config['calib_dataset']['batch_size'] = args.calib_batch_size
    if args.input_calib_type:
        config['calib']['input_type'] = args.input_calib_type
    if args.weight_calib_type:
        config['calib']['weight_type'] = args.weight_calib_type

    return config

def resume_train(resume, model, optimizer, device, cnmix):
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location=torch.device('cpu'))
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model_state = checkpoint['model']
        remove_module_prefix(model_state)
        model.load_state_dict(model_state, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if cnmix and isinstance(checkpoint, dict) and 'cnmix' in checkpoint:
            import cnmix
            cnmix.load_state_dict(checkpoint['cnmix'])
        print("=> loaded checkpoint '{}' (epoch {})"
            .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    return start_epoch, best_acc1


def save_ckpt(state, is_best, path='.', epoch=None):
    if not os.path.exists(path):
        os.mkdir(path)
    if epoch:
        filename = os.path.join(path, "epoch_{}.pth".format(epoch))
    else:
        filename = os.path.join(path, 'last.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'best.pth'))


def set_device(dev=None, use_cpu=True, use_device='cuda'):
    r"""Set the device

    Args:
        gpu (int, default=None): gpu id to use
        use_cpu (bool, default=True): use cpu device

    Returns:
        device (torch.device)
    """

    if use_cpu:
        device = torch.device("cpu")
    elif use_device == 'mlu':
        if dev is not None:
            import torch_mlu
            import torch_mlu.core.mlu_model as ct
            ct.set_device(dev)
            device = torch.device("mlu:{}".format(dev))
        else:
            device = torch.device("mlu:0")
    elif use_device == 'cuda':
        if dev is not None:
            torch.cuda.set_device(dev)
            device = torch.device("cuda:{}".format(dev))
        else:
            device = torch.device("cuda:0")

    return device


def set_writer(prefix='logs', rank=0, total=1):
    r"""Set the writer

    Args:
        prefix (str, default='logs'): tensorboardX writer path
        rank (int, default=0): log rank
        total (int, default=1): log total nodes

    Returns:
        writer (SummaryWriter)
    """

    writer = SummaryWriter('{}/{}_{}'.format(prefix, rank, total))

    return writer

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    from PIL import Image
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageCalibDataset(torch.utils.data.Dataset):
    def __init__(self, root, filename, loader=default_loader, transform=None):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.samples = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.samples.append(
                    os.path.join(self.root, line.strip()))

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)


def create_calib_loader(data_config, loader_config):
    calib_transform = create_transform(data_config['transform'])
    calib_dataset = ImageCalibDataset(data_config['path'], data_config['file'],
        transform=calib_transform)
    calib_loader = torch.utils.data.DataLoader(
        calib_dataset, batch_size=loader_config['batch_size'],
        shuffle=loader_config['shuffle'],
        num_workers=loader_config['workers'],
        pin_memory=loader_config['pin_memory'])

    return calib_loader
