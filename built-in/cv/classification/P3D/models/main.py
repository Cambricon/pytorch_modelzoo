# from __future__ import print_function
import argparse
import torch
from train import Training
from logger import Logger
import os
import random
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Pseudo-3D fine-tuning')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--data-set', default='UCF101', const='UCF101', nargs='?', choices=['UCF101', 'Breakfast', 'merl'])
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--early-stop', default=10, type=int, metavar='N', help='number of early stopping')
parser.add_argument('--epochs', default=75, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--dropout', default=0.5, type=float, metavar='M', help='dropout')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--evaluate', default='', type=str, metavar='PATH', help='path of checkpoint that used to evaluate model on validation set')
parser.add_argument('--random', dest='random', action='store_true', help='random pick image')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--model-type', default='P3D', choices=['P3D', 'C3D', 'I3D'], help='which model to run the code')
parser.add_argument('--num-frames', default=16, type=int, metavar='N', help='number frames per clip')
parser.add_argument('--logdir',type=str,default='./',metavar='DIR', help='Where to save logs')
parser.add_argument('--train_steps', default=-1, type=int,
                    help='how many iterations to train.')
parser.add_argument('--eval_steps', default=-1, type=int,
                    help='how many iterations to evaluate.')
parser.add_argument('--num-dev', type=int, default=4,
                    help='Number of GPUS to use')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--cnmix', action='store_true', default=False,
                    help='use cnmix for mixed precision training')
parser.add_argument('--pyamp', action='store_true', default=False,
                    help='use pytorch amp for mixed precision training')
parser.add_argument('--opt_level', type=str, default='O1',
                    help='choose level of mixing precision')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--device_param", type=str, default="mlu",
                    choices=["mlu","gpu","cpu"], help="device to run the model")
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--dummy_test', dest='dummy_test', action='store_true',
                        help='use fake data to traing')

try:
    import torch_mlu.core.mlu_model as ct
except ImportError:
    print("without torch_mlu")

def main():
    args = parser.parse_args()
    args = vars(args)

    if args['resume'] and args['evaluate']:
        print("invalid args: resume and evaluate can not be set at the same time!")
        exit()

    if args['data_set'] == 'UCF101':
        print('UCF101 data set')
        name_list = 'ucfTrainTestlist'
        num_classes = 101
    elif args['data_set'] == 'Breakfast':
        print("breakfast data set")
        num_classes = 37
        name_list = 'breakfastTrainTestList'
    else:
        print('Merl data set')
        num_classes = 5
        name_list = 'merlTrainTestList'

    args['distributed'] = False
    if 'WORLD_SIZE' in os.environ:
        args['distributed'] = int(os.environ['WORLD_SIZE']) > 1
    if args['device_param'] == "mlu":
        args['device'] = 'mlu:0'
    else:
        args['device'] = 'cuda:0'
    args['world_size'] = 1

    args['rank']=args['local_rank']

    if args['distributed']:
        args['num_dev'] = 1
        if args['device_param'] == "mlu":
            args['device'] = 'mlu:%d' % args['local_rank']
            ct.set_device(args['local_rank'])
        else:
            args['device'] = 'cuda:%d' % args['local_rank']
            torch.cuda.set_device(args['local_rank'])
        torch.distributed.init_process_group(backend=args['dist_backend'], init_method='env://')
        args['world_size'] = torch.distributed.get_world_size()
        args['rank'] = torch.distributed.get_rank()
        print('Training in distributed mode with multiple processes, 1 %s per process. Process %d, total %d.'
                     % (args['device_param'], args['rank'], args['world_size']))
    else:
        print('Training with a single process on %d %s.' % (args['num_dev'], args['device_param']))

    torch.manual_seed(args['seed'])
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed']) # if you are using multi-GPU.

    Training(name_list=name_list, num_classes=num_classes, modality='RGB', **args)


if __name__ == '__main__':
    main()
