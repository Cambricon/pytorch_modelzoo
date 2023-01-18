import os
import argparse
import pprint
from data import dataloader_imagenet
from run_networks import model
import warnings
from utils import source_import, preprocess_train_config
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import torch_mlu.core.mlu_model as ct
    _USE_MLU = True
except ImportError:
    _USE_MLU = False

# ================
# LOAD CONFIGURATIONS
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='brand_mid_stage_1.py', type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
# parser.add_argument('--attri', default='brand_mid')
parser.add_argument('--device', default='0')
parser.add_argument('--loadertype', default='')
parser.add_argument("--data_path",  default='../data/vehicle_brand',
        help="input dataset path (i.e., directory of dataset)")
parser.add_argument("--resume", default = None, type=str,
        help="path to latest checkpoint (default: none)")
parser.add_argument("--iters", type=int, default=-1, metavar='N', help="run N iters")
parser.add_argument('--seed', default=1, type=int,
        help='seed for initializing training. ')
parser.add_argument('--cnmix', action='store_true', default=False,
        help='use cnmix for mixed precision training')
parser.add_argument('--opt_level', type=str, default='O1',
        help='choose level of mixing precision')
parser.add_argument('--dummy_test', dest='dummy_test', action='store_true',
                        help='use fake data to traing')
parser.add_argument('--num_workers', default=8, type=int,
        help='num_workers for dataloader. ')
parser.add_argument('--batch_size', default=128, type=int,
        help='batch_size for training. ')

args = parser.parse_args()
loadertype = args.loadertype
if _USE_MLU:
    os.environ["MLU_VISIBLE_DEVICES"] = args.device
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
data_root = {'ImageNet': 'data/ImageNet_LT/'}
data_dir = args.data_path
test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits
configpath = 'config/ImageNet_LT/'+args.config
config = source_import(configpath).config
config = preprocess_train_config(args, config)
training_opt = config['training_opt']
# change
relatin_opt = config['memory']
dataset = training_opt['dataset']
checkpoint = os.path.join(training_opt['log_dir'], 'final_model_checkpoint.pth')
# add for precheckin, daily and benchmark, which iters is not -1
if args.iters != -1:
    checkpoint = args.resume
    config['training_opt']['num_epochs']=1

if loadertype != '':
    training_opt['log_dir'] += ('_train'+loadertype)
    if 'weightpath' in config['networks']['feat_model']['params']:
        config['networks']['feat_model']['params']['weightpath'] += ('_train'+loadertype)
        config['networks']['classifier']['params']['weightpath'] += ('_train'+loadertype)

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

print('Loading dataset from: %s' % data_root[dataset.rstrip('_LT')])
pprint.pprint(config)

def wrapper(training_model, args, dev_id, data):
    tmp = 0
    for key, val in training_model.networks.items():
        if _USE_MLU and args.cnmix:
            import cnmix
            optimizer = training_model.model_optimizer if tmp == 0 and not test_mode else None
            training_model.networks[key], optimizer = cnmix.initialize(training_model.networks[key], optimizer, opt_level=args.opt_level)
            cnmix.cnmix_set_amp_quantify_params('all', {'batch_size': config['training_opt']['batch_size'], 'data_num': len(data['train'].dataset)})
            tmp += 1
        training_model.networks[key] = DDP(training_model.networks[key], device_ids=[dev_id])

def main():
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    ndevs_per_node = args.device.count(',') + 1
    mp.spawn(main_worker, nprocs=ndevs_per_node, args=(ndevs_per_node, args))
    #main_worker(args.device, ndevs_per_node, args)


def main_worker(dev_id, ndevs_per_node, args):
    dev_id = int(dev_id)
    if _USE_MLU:
        backend = "cncl"
        ct.set_device(dev_id)
    else:
        backend = "nccl"
        torch.cuda.set_device(dev_id)
    dist.init_process_group(backend=backend, init_method="env://", rank=dev_id, world_size=ndevs_per_node)

    if not test_mode:
        sampler_defs = training_opt['sampler']
        if sampler_defs:
            sampler_dic = {'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                           'num_samples_cls': sampler_defs['num_samples_cls']}
        else:
            sampler_dic = None

        data = {x: dataloader_imagenet.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                                 data_dir=data_dir,
                                                 dataset=dataset,
                                                 phase=x,
                                                 batch_size=training_opt['batch_size'],
                                                 sampler_dic=sampler_dic,
                                                 num_workers=training_opt['num_workers'],
            loadertype=loadertype) for x in (['train', 'val', 'train_plain'] if relatin_opt['init_centroids'] else ['train', 'val'])}

        training_model = model(args, ndevs_per_node, dev_id, config, data, test=False)
        wrapper(training_model, args, dev_id, data)
        if args.resume:
            training_model.load_model(checkpoint)


        training_model.train()

    else:

        warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

        print('Under testing phase, we load training data simply to calculate training data number for each class.')

        data = {
                x: dataloader_imagenet.load_data(data_root=data_root[dataset.rstrip('_LT')],
                data_dir=data_dir,
                dataset=dataset,
                phase=x,
                batch_size=training_opt['batch_size'],
                sampler_dic=None,
                test_open=test_open,
                num_workers=training_opt['num_workers'],
                shuffle=False,
                loadertype=loadertype) for x in ['train', 'test']}

        training_model = model(args, ndevs_per_node, dev_id, config, data, test=True)
        wrapper(training_model, args, dev_id, data)
        training_model.load_model(checkpoint)
        training_model.eval(phase='test', openset=test_open)

        if output_logits:
            training_model.output_logits(openset=test_open)

    print('ALL COMPLETED.')

if __name__ == '__main__':
    start_time=time.time()
    main()
    end_time=time.time()
    print("use time: ", end_time-start_time)
