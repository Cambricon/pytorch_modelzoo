import torch.jit
import os
import sys
import gc
import math
import copy
import re
import time
import timeit
from datetime import datetime
from collections import OrderedDict
from argparse import ArgumentParser
from alias_generator import AliasSample
import pickle
from convert import generate_negatives
from convert import generate_negatives_flat
from convert import CACHE_FN

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn

import utils
from neumf import NeuMF

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../../tools/utils/")
print(cur_dir + "/../../../../../tools/utils/")
from metric import MetricCollector

from mlperf_compliance import mlperf_log
def parse_args():
    parser = ArgumentParser(description="Train a Nerual Collaborative"
                                        " Filtering model")
    parser.add_argument('--data', type=str,
                        help='path to test and training data files')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='number of epochs for training')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='number of examples for each iteration')
    parser.add_argument('--valid-batch-size', type=int, default=2**20,
                        help='number of examples in each validation chunk')
    parser.add_argument('-f', '--factors', type=int, default=8,
                        help='number of predictive factors')
    parser.add_argument('--layers', nargs='+', type=int,
                        default=[64, 32, 16, 8],
                        help='size of hidden layers for MLP')
    parser.add_argument('-n', '--negative-samples', type=int, default=4,
                        help='number of negative examples per interaction')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
                        help='learning rate for optimizer')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='rank for test examples to be considered a hit')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float,
                        help='stop training early at threshold')
    parser.add_argument('--valid-negative', type=int, default=999,
                        help='Number of negative samples for each positive test example')
    parser.add_argument('--processes', '-p', type=int, default=1,
                        help='Number of processes for evaluating model')
    parser.add_argument('--workers', '-w', type=int, default=8,
                        help='Number of workers for training DataLoader')
    parser.add_argument('--beta1', '-b1', type=float, default=0.9,
                        help='beta1 for Adam')
    parser.add_argument('--beta2', '-b2', type=float, default=0.999,
                        help='beta1 for Adam')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='eps for Adam')
    parser.add_argument('--user_scaling', default=1, type=int)
    parser.add_argument('--item_scaling', default=1, type=int)
    parser.add_argument('--cpu_dataloader', action='store_true',
                        help='pre-process data on cpu to save memory')
    parser.add_argument('--random_negatives', action='store_true',
                        help='do not check train negatives for existence in dataset')
    parser.add_argument('--iters', type=int, default=30000, metavar='N',
                        help='iters per epoch')
    parser.add_argument('--inference-iters', type=int, default=300000, metavar='N',
                        help='iters for inference')
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--device', default='cpu', type=str,
                        help='Use cpu gpu or mlu device')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--save_ckp", dest='save_ckp', type=int,
                            help="Enable save checkpoint")
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--ckpdir',type=str,default='./ckps',metavar='DIR',
                        help='Where to save ckps')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--use_amp', default='0', type=int,
                    help='use Automatic Mixed Precision')
    parser.add_argument('--dummy_test', default=False, action='store_true', help='use dummy test for benchmark')
    return parser.parse_args()


class dummy_data_loader():
    # MovieLens: user, [B], item, [B], label, [B]
    def __init__(self, len = 0, batch_size = 65536, length = 100):
        self.len = len
        self.user = torch.randint(low=0, high=65536, size = [batch_size], dtype=torch.long)
        self.item = torch.randint(low=0, high=65536, size = [batch_size], dtype=torch.long)
        self.label = torch.randn(batch_size)
        self.data = 0
    def __iter__(self):
        return self
    def __len__(self):
        return self.len
    def __next__(self):
        if self.data > self.len:
            raise StopIteration
        else:
           self.data+=1
           return self.user, self.item, self.label
    def next(self):
        return self.__next__()

# TODO: val_epoch is not currently supported on cpu
def val_epoch(model, args, num_user, output=None, epoch=None):
    if args.device == 'mlu':
        import torch_mlu.core.mlu_model as ct
    device = ''
    if args.device == 'gpu':
        device = 'cuda'
    elif args.device == 'mlu':
        device = 'mlu'
    nb_users = num_user
    print(datetime.now(), "Loading test ratings.")
    test_ratings = [torch.LongTensor()] * args.user_scaling

    for chunk in range(args.user_scaling):
        test_ratings[chunk] = torch.from_numpy(np.load(args.data + '/testx' 
                + str(args.user_scaling) + 'x' + str(args.item_scaling) 
                + '_' + str(chunk) + '.npz', encoding='bytes')['arr_0'])
    
    test_pos = [l[:,1].reshape(-1,1) for l in test_ratings]
    test_negatives = [torch.LongTensor()] * args.user_scaling
    test_neg_items = [torch.LongTensor()] * args.user_scaling
    
    print(datetime.now(), "Loading test negatives.")
    for chunk in range(args.user_scaling):
        file_name = (args.data + '/test_negx' + str(args.user_scaling) + 'x'
                + str(args.item_scaling) + '_' + str(chunk) + '.npz')
        raw_data = np.load(file_name, encoding='bytes')
        test_negatives[chunk] = torch.from_numpy(raw_data['arr_0'])
        print(datetime.now(), "Test negative chunk {} of {} loaded ({} users).".format(
            chunk+1, args.user_scaling, test_negatives[chunk].size()))

    test_neg_items = [l[:, 1] for l in test_negatives]

    # create items with real sample at last position
    test_items = [torch.cat((a.reshape(-1,args.valid_negative), b), dim=1)
            for a, b in zip(test_neg_items, test_pos)]
    del test_ratings, test_neg_items
    # generate dup mask and real indice for exact same behavior on duplication compare to reference
    # here we need a sort that is stable(keep order of duplicates)
    # this is a version works on integer
    sorted_items, indices = zip(*[torch.sort(l) for l in test_items]) # [1,1,1,2], [3,1,0,2]
    sum_item_indices = [a.float()+b.float()/len(b[0]) 
            for a, b in zip(sorted_items, indices)] #[1.75,1.25,1.0,2.5]
    indices_order = [torch.sort(l)[1] for l in sum_item_indices] #[2,1,0,3]
    stable_indices = [torch.gather(a, 1, b) 
            for a, b in zip(indices, indices_order)] #[0,1,3,2]
    # produce -1 mask
    dup_mask = [(l[:,0:-1] == l[:,1:]) for l in sorted_items]
    dup_mask = [torch.cat((torch.zeros_like(a, dtype=torch.uint8), b),dim=1)
            for a, b in zip(test_pos, dup_mask)]
    dup_mask = [torch.gather(a,1,b.sort()[1])
            for a, b in zip(dup_mask, stable_indices)]
    # produce real sample indices to later check in topk
    sorted_items, indices = zip(*[(a != b).float().sort()
            for a, b in zip(test_items, test_pos)])
    sum_item_indices = [(a.float()) + (b.float())/len(b[0])
            for a, b in zip(sorted_items, indices)]
    indices_order = [torch.sort(l)[1] for l in sum_item_indices]
    stable_indices = [torch.gather(a, 1, b)
            for a, b in zip(indices, indices_order)]
    real_indices = [l[:, 0] for l in stable_indices]
    del sorted_items, indices, sum_item_indices, indices_order, stable_indices, test_pos

    # For our dataset, test set is identical to user set, so arange() provides
    # all test users.
    test_users = torch.arange(nb_users, dtype=torch.long)
    test_users = test_users[:, None]
    test_users = test_users + torch.zeros(1+args.valid_negative, dtype=torch.long)
    # test_items needs to be of type Long in order to be used in embedding
    test_items = torch.cat(test_items).type(torch.long)

    dup_mask = torch.cat(dup_mask)
    real_indices = torch.cat(real_indices)

    # make pytorch memory behavior more consistent later
    if device == 'cuda':
        torch.cuda.empty_cache()
    if args.device == 'mlu':
        torch.mlu.empty_cache()

    mlperf_log.ncf_print(key=mlperf_log.INPUT_BATCH_SIZE, value=args.batch_size)
    mlperf_log.ncf_print(key=mlperf_log.INPUT_ORDER)  # we shuffled later with randperm

    # Calculate initial Hit Ratio and NDCG
    samples_per_user = test_items.size(1)
    users_per_valid_batch = args.valid_batch_size // samples_per_user

    test_users = test_users.split(users_per_valid_batch)
    test_items = test_items.split(users_per_valid_batch)
    dup_mask = dup_mask.split(users_per_valid_batch)
    real_indices = real_indices.split(users_per_valid_batch)
    
    start = datetime.now()
    log_2 = math.log(2)

    model.to(device)
    model.eval()
    ndcg = torch.tensor(0., device=device)
    hits = torch.tensor(0., device=device)

    K = args.topk
    with torch.no_grad():
        for i, (u,n) in enumerate(zip(test_users,test_items)):
            if args.device == 'gpu':
                res = model(u.cuda().view(-1), n.cuda().view(-1), sigmoid=True).detach().view(-1,samples_per_user)
            if args.device == 'mlu':
                res = model(u.to('mlu').view(-1), n.to('mlu').view(-1), sigmoid=True).detach().view(-1,samples_per_user)
            # set duplicate results for the same item to -1 before topk
            res[dup_mask[i].to(torch.bool)] = -1
            out = torch.topk(res,K)[1]
            # topk in pytorch is stable(if not sort)
            # key(item):value(predicetion) pairs are ordered as original key(item) order
            # so we need the first position of real item(stored in real_indices) to check if it is in topk
            if args.device == 'gpu':
                ifzero = (out == real_indices[i].cuda().view(-1,1))
            if args.device == 'mlu':
                ifzero = (out == real_indices[i].to('mlu').view(-1,1))
            hits += ifzero.sum()
            ndcg += (log_2 / (torch.nonzero(ifzero, as_tuple=False)[:,1].view(-1).to(torch.float)+2).log_()).sum()
            if i == args.inference_iters:
                break

    mlperf_log.ncf_print(key=mlperf_log.EVAL_SIZE, value={"epoch": epoch, "value": num_user * samples_per_user})
    mlperf_log.ncf_print(key=mlperf_log.EVAL_HP_NUM_USERS, value=num_user)
    mlperf_log.ncf_print(key=mlperf_log.EVAL_HP_NUM_NEG, value=samples_per_user - 1)

    end = datetime.now()

    hits = hits.item()
    ndcg = ndcg.item()

    if output is not None:
        result = OrderedDict()
        result['timestamp'] = datetime.now()
        result['duration'] = end - start
        result['epoch'] = epoch
        result['K'] = K
        result['hit_rate'] = hits/num_user
        result['NDCG'] = ndcg/num_user
        if os.path.exists(output):
            utils.save_result(result, output)

    return hits/num_user, ndcg/num_user


def main():

    args = parse_args()
    if args.device == 'mlu':
        import torch_mlu.core.mlu_model as ct
    if args.use_amp:
        from torch.cuda.amp import autocast, GradScaler
    args.ndevices=1
    if args.device == 'mlu':
        args.ndevices = ct.device_count()
    if args.device == 'gpu':
        args.ndevices = torch.cuda.device_count()

    args.distributed = args.multiprocessing_distributed  or args.world_size > 1
    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and args.device == "gpu" else args.device)
        n_device = 1
    else:
        if args.device == "mlu":
            ct.set_device(args.local_rank)
        else:
            torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda" if torch.cuda.is_available() and args.device == "gpu" else args.device)
        torch.distributed.init_process_group(backend='cncl' if args.device == "mlu" else "nccl", init_method='env://')
        n_device = 1

    print("device: {} n_device: {}".format(device, n_device))

    # Save configuration to file
    config = {k: v for k, v in args.__dict__.items()}
    config['timestamp'] = "{:.0f}".format(datetime.utcnow().timestamp())
    config['local_timestamp'] = str(datetime.now())
    run_dir = "./run/neumf/{}".format(config['timestamp'])
    if args.local_rank == -1:
        print("Saving config and results to {}".format(run_dir))
        if not os.path.exists(run_dir) and run_dir != '':
            os.makedirs(run_dir)
        utils.save_config(config, run_dir)

    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # Check where to put data loader
    if use_cuda:
        dataloader_device = 'cpu' if args.cpu_dataloader else 'cuda'
    else:
        dataloader_device = 'cpu'

    # more like load trigger timmer now
    mlperf_log.ncf_print(key=mlperf_log.PREPROC_HP_NUM_EVAL, value=args.valid_negative)
    # The default of np.random.choice is replace=True, so does pytorch random_()
    mlperf_log.ncf_print(key=mlperf_log.PREPROC_HP_SAMPLE_EVAL_REPLACEMENT, value=True)
    mlperf_log.ncf_print(key=mlperf_log.INPUT_HP_SAMPLE_TRAIN_REPLACEMENT, value=True)
    mlperf_log.ncf_print(key=mlperf_log.INPUT_STEP_EVAL_NEG_GEN)

    # sync worker before timing.
    if use_cuda:
        torch.cuda.synchronize()

    if args.do_train:
        #===========================================================================
        #== The clock starts on loading the preprocessed data. =====================
        #===========================================================================
        mlperf_log.ncf_print(key=mlperf_log.RUN_START)
        run_start_time = time.time()

        fn_prefix = args.data + '/' + CACHE_FN.format(args.user_scaling, args.item_scaling)
        sampler_cache = fn_prefix + "cached_sampler.pkl"
        print(datetime.now(), "Loading preprocessed sampler.")
        sampler = None
        pos_users = None
        pos_items = None
        if os.path.exists(args.data):
            print("Using alias file: {}".format(args.data))
        with open(sampler_cache, "rb") as f:
            sampler, pos_users, pos_items, nb_items, _ = pickle.load(f)
        
        print(datetime.now(), "Alias table loaded.")
        train_users = None
        train_items = None
        if args.distributed:
            if args.local_rank != 0:
                # Make sure only the first process in distributed training process the dataset, and the others will use the cache
                torch.distributed.barrier()
            cached_features_files = []
            cache_file_prefix = os.getcwd() +'/' + 'data' + '/'
            for i in range(args.ndevices):
                cached_features_files.append(os.path.join(
                    cache_file_prefix ,
                    "cached_{}_{}_{}".format(
                        "dlrm",
                        i,
                        args.ndevices
                    ),
                ))

            if not os.path.exists(cache_file_prefix):
                os.makedirs(cache_file_prefix)

            if os.path.exists(cached_features_files[args.local_rank]):
                train_users_items = torch.load(cached_features_files[args.local_rank])
                train_users, train_items = (
                    train_users_items["train_users"],
                    train_users_items["train_items"],
                )
            else:
                print("dividing dataset into {} part".format(args.ndevices))
                pos_users_list = np.array_split(pos_users, args.ndevices)
                pos_items_list = np.array_split(pos_items, args.ndevices)
                for i in range(args.ndevices):
                    torch.save({"train_users": 
                                torch.tensor([f for f in pos_users_list[i]],
                                                dtype=torch.long),
                                "train_items":
                                torch.tensor([f for f in pos_items_list[i]],
                                                dtype=torch.long)
                                }, cached_features_files[i])
                train_users = torch.tensor([f for f in pos_users_list[0]], dtype=torch.long)
                train_items = torch.tensor([f for f in pos_items_list[0]], dtype=torch.long)
            if args.local_rank == 0:
                torch.distributed.barrier()
        
        nb_users = len(sampler.num_regions)
        # train_users = torch.from_numpy(pos_users).type(torch.LongTensor)
        # train_items = torch.from_numpy(pos_items).type(torch.LongTensor)
        if not args.distributed:
            train_users = torch.tensor([f for f in pos_users], dtype=torch.long)
            train_items = torch.tensor([f for f in pos_items], dtype=torch.long)
        # train_data = TensorDataset(train_users, train_items)

        del sampler, pos_users, pos_items
        mlperf_log.ncf_print(key=mlperf_log.INPUT_SIZE, value=len(train_users))
        # produce things not change between epoch
        # mask for filtering duplicates with real sample
        # note: test data is removed before create mask, same as reference
        # create label
        train_label = torch.ones_like(train_users, dtype=torch.float32)
        neg_label = torch.zeros_like(train_label, dtype=torch.float32)
        neg_label = neg_label.repeat(args.negative_samples)
        train_label = torch.cat((train_label,neg_label))
        print(datetime.now(),
            "Train Data loading done {:.1f} sec. #user={}, #item={}, #train={}, #test={}".format(
            time.time()-run_start_time, nb_users, nb_items, len(train_users), nb_users))
        del neg_label

        gc.collect(generation=2)

        # Create model
        model = NeuMF(nb_users, nb_items,
                    mf_dim=args.factors, mf_reg=0.,
                    mlp_layer_sizes=args.layers,
                    mlp_layer_regs=[0. for i in args.layers])
        print(model)
        print("{} parameters".format(utils.count_parameters(model)))
        if args.use_amp:
            scaler = GradScaler()

        mlperf_log.ncf_print(key=mlperf_log.OPT_LR, value=args.learning_rate)
        mlperf_log.ncf_print(key=mlperf_log.OPT_NAME, value="Adam")
        mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_BETA1, value=args.beta1)
        mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_BETA2, value=args.beta2)
        mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_EPSILON, value=args.eps)
        mlperf_log.ncf_print(key=mlperf_log.MODEL_HP_LOSS_FN, value=mlperf_log.BCE)

        # Add optimizer and loss to graph
        params = model.parameters()

        criterion = nn.BCEWithLogitsLoss(reduction = 'none') # use torch.mean() with dim later to avoid copy to host
        if use_cuda:
        # Move model and loss to GPU
            model = model.cuda()
            criterion = criterion.cuda()
        if args.device == 'mlu':
            model = model.to('mlu')
            criterion = criterion.to('mlu')
        local_batch = args.batch_size
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps)
        traced_criterion = torch.jit.trace(criterion.forward, (torch.rand(local_batch,1),torch.rand(local_batch,1)))
        args.start_epoch = 0
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> Loading checkpoint: {}".format(args.resume))
                resume_point = torch.load(args.resume, map_location=torch.device('cpu'))
                resume_point_replace = {}
                args.start_epoch = resume_point['epoch']
                print("Resume from epoch {}".format(args.start_epoch))
                # Remove "submodule" (e.g model.submodule.conv1 -> model.conv1)
                # and "module" (e.g features.module.conv2d -> features.conv2d)
                # they are created during DDP training, different from origin model
                for key in list(resume_point['state_dict'].keys()):
                    split_key = key.split('.')
                    split_origin = copy.deepcopy(split_key)
                    for item in split_origin:
                        if item == "module":
                            split_key.remove("module")
                    resume_point_replace[".".join(split_key)] = resume_point['state_dict'][key]
                model.load_state_dict(resume_point_replace, strict=True if args.device=='gpu' else False)
                if args.use_amp :
                    try:
                        scaler.load_state_dict(resume_point['amp'])
                    except:
                        print("warning no amp in ckp")
                del resume_point
                del resume_point_replace
                gc.collect(generation=2)
            else:
                print("ERROR: Fail to load Resume checkpoint from {}, file not exist".format(args.resume))
                return
            
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        elif n_device > 1:
            model = torch.nn.DataParallel(model)
        # success = False
        mlperf_log.ncf_print(key=mlperf_log.TRAIN_LOOP)

        adaptive_cnt = int(os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT')) if (os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT') is not None) else 0
        batch_time_m = AverageMeter('BatchTimeAve', ':6.3f')
        data_time_m = AverageMeter('DataTimeAve', ':6.3f')
        losses_m = AverageMeter('Loss', ':6.3f')

        for epoch in range(args.start_epoch, args.epochs):
            
            mlperf_log.ncf_print(key=mlperf_log.TRAIN_EPOCH, value=epoch)
            mlperf_log.ncf_print(key=mlperf_log.INPUT_HP_NUM_NEG, value=args.negative_samples)
            mlperf_log.ncf_print(key=mlperf_log.INPUT_STEP_TRAIN_NEG_GEN)
            begin = time.time()
            
            st = timeit.default_timer()
            
            if args.random_negatives:
                neg_users = train_users.repeat(args.negative_samples)
                neg_items = torch.empty_like(neg_users, dtype=torch.int64).random_(0, nb_items)
            else:
                negatives = generate_negatives(
                    sampler,
                    args.negative_samples,
                    train_users.numpy())
                negatives = torch.from_numpy(negatives)
                neg_users = negatives[:, 0]
                neg_items = negatives[:, 1]
            epoch_users = torch.cat((train_users,neg_users))
            epoch_items = torch.cat((train_items,neg_items))
            train_data = TensorDataset(epoch_users, epoch_items, train_label)

            train_sampler = RandomSampler(train_data)

            train_dataloader = DataLoader(train_data,
                                            sampler=train_sampler,
                                            batch_size=local_batch * n_device,
                                            num_workers=args.workers,
                                            pin_memory=True)
            print("generate_negatives loop time: {:.2f}".format(timeit.default_timer() - st))
            after_neg_gen = time.time()

            st = timeit.default_timer()

            train_iter = tqdm(train_dataloader)

            if args.dummy_test:
                train_iter = tqdm(dummy_data_loader(batch_size=args.batch_size, len=len(train_iter)))
            
            after_shuffle = time.time()

            neg_gen_time = (after_neg_gen - begin)
            shuffle_time = (after_shuffle - after_neg_gen)
            end_time = time.time()

            # for internal benchmark test
            metric_collector = MetricCollector(
                enable_only_benchmark=True,
                record_elapsed_time=True,
                record_hardware_time=True if args.device == 'mlu' else False)
            metric_collector.place()

            for i, batch in enumerate(train_iter):
                if i == args.iters:
                    break
                data_time_m.update(time.time() - end_time)
                if n_device == 1:
                    batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                user, item, label = batch
                label = label.view(-1, 1)

                for p in model.parameters():
                    p.grad = None
                if args.use_amp : 
                    with autocast():
                        outputs = model(user, item)
                        loss = traced_criterion(outputs, label).float()
                        loss = torch.mean(loss.view(-1), 0)
                else:
                    outputs = model(user, item)
                    loss = traced_criterion(outputs, label).float()
                    loss = torch.mean(loss.view(-1), 0)
                if n_device > 1:
                    loss = loss.mean()
                
                if args.use_amp :
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                train_iter.set_postfix(loss='{:05.3f}'.format(loss.item()))
                
                metric_collector.record()
                metric_collector.place()

                losses_m.update(loss.item(), user.size(0))
                batch_time_m.update(time.time() - end_time)
                end_time = time.time()

            metric_collector.insert_metrics(
                net = "DLRM",
                batch_size = args.batch_size,
                precision = "amp" if args.use_amp else "fp32",
                cards = ct.device_count() if args.local_rank == 0 else 1,
                DPF_mode = "ddp " if args.multiprocessing_distributed == True else "single")
            if ((args.distributed == False and args.local_rank == -1) or (args.local_rank == 0)):
                metric_collector.dump()

            if args.save_ckp == 1:
                if args.device == 'mlu':
                    if (args.distributed == False) or (args.local_rank == 0): # Only save checkpoint by Process 0
                        if not os.path.exists(args.ckpdir):
                            os.makedirs(args.ckpdir)
                        if args.use_amp :
                            save_file_path = os.path.join(args.ckpdir, "dlrm" + "amp" + "_" + str(epoch) + ".pth")
                        else:
                            save_file_path = os.path.join(args.ckpdir, "dlrm" + "_" + str(epoch) + ".pth")
                        print("=> Save file to {}".format(save_file_path))
                        if args.use_amp:
                            checkpoint = {"state_dict":model.state_dict(), "optimizer":optimizer.state_dict(),
                                            "epoch": epoch, "amp":scaler.state_dict()}
                        else:
                            checkpoint = {"state_dict":model.state_dict(), "optimizer":optimizer.state_dict(),
                                            "epoch": epoch}
                        # if args.use_amp:
                        #     checkpoint["amp"]=amp.state_dict()
                        torch.save(checkpoint, save_file_path)
                        print("=> Model save finished")

                        # Load from ckp:
                elif args.device == 'gpu':
                    if args.distributed == False or args.local_rank == 0:
                        if not os.path.exists(args.ckpdir):
                            os.makedirs(args.ckpdir)
                        if args.use_amp : 
                            save_file_path = os.path.join(args.ckpdir, "dlrm" + "amp" + "_" + str(epoch) + ".pth")
                        else:
                            save_file_path = os.path.join(args.ckpdir, "dlrm" + "_" + str(epoch) + ".pth")
                        print("=> Save file to {}".format(save_file_path))
                        if args.use_amp : 
                            checkpoint = {"state_dict":model.state_dict(), "optimizer":optimizer.state_dict(),
                                            "epoch": epoch, "amp":scaler.state_dict()}
                        else:
                            checkpoint = {"state_dict":model.state_dict(), "optimizer":optimizer.state_dict(),
                                            "epoch": epoch}
                        torch.save(checkpoint, save_file_path)
                        print("=> Model save finished")
                
            if args.iters <= len(train_dataloader) and args.iters != -1:
                break
            train_time = time.time() - begin

        mlperf_log.ncf_print(key=mlperf_log.RUN_FINAL)
    
    if args.do_predict and ( args.local_rank == 0 or args.local_rank == -1) :
        run_start_time = time.time()
        fn_prefix = args.data + '/' + CACHE_FN.format(args.user_scaling, args.item_scaling)
        sampler_cache = fn_prefix + "cached_sampler.pkl"
        print(datetime.now(), "Loading preprocessed sampler.")
        if os.path.exists(args.data):
            print("Using alias file: {}".format(args.data))
        with open(sampler_cache, "rb") as f:
            sampler, pos_users, pos_items, nb_items, _ = pickle.load(f)
        print(datetime.now(), "Alias table loaded.")

        nb_users = len(sampler.num_regions)
        train_users = torch.tensor([f for f in pos_users], dtype=torch.long)
        train_items = torch.tensor([f for f in pos_items], dtype=torch.long)
        del sampler, pos_users, pos_items
        mlperf_log.ncf_print(key=mlperf_log.INPUT_SIZE, value=len(train_users))
        train_label = torch.ones_like(train_users, dtype=torch.float32)
        neg_label = torch.zeros_like(train_label, dtype=torch.float32)
        neg_label = neg_label.repeat(args.negative_samples)
        train_label = torch.cat((train_label,neg_label))
        print(datetime.now(),
            "Train Data loading done {:.1f} sec. #user={}, #item={}, #train={}, #test={}".format(
            time.time()-run_start_time, nb_users, nb_items, len(train_users), nb_users))

        # Create model
        model = NeuMF(nb_users, nb_items,
                    mf_dim=args.factors, mf_reg=0.,
                    mlp_layer_sizes=args.layers,
                    mlp_layer_regs=[0. for i in args.layers])
        print(model)
        print("{} parameters".format(utils.count_parameters(model)))

        mlperf_log.ncf_print(key=mlperf_log.OPT_LR, value=args.learning_rate)
        mlperf_log.ncf_print(key=mlperf_log.OPT_NAME, value="Adam")
        mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_BETA1, value=args.beta1)
        mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_BETA2, value=args.beta2)
        mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_EPSILON, value=args.eps)
        mlperf_log.ncf_print(key=mlperf_log.MODEL_HP_LOSS_FN, value=mlperf_log.BCE)
        if os.path.isfile(args.resume):
            if os.path.exists(args.ckpdir) and len(os.listdir(args.ckpdir)) > 0:
                if os.path.isfile(args.ckpdir + '/' +'dlrmamp_' + str(args.start_epoch) + '.pth'):
                    args.resume = args.ckpdir + '/' +'dlrmamp_' + str(args.start_epoch) + '.pth'
                else:
                    args.resume = args.ckpdir + '/' +'dlrm_' + str(args.start_epoch) + '.pth'
            print("=> Loading checkpoint: {}".format(args.resume))
            resume_point = torch.load(args.resume, map_location=torch.device('cpu'))
            resume_point_replace = {}
            args.start_epoch = resume_point['epoch']
            print("Resume from epoch {}".format(args.start_epoch))
            if args.distributed: # DDP module create by multi device
                # Remove "submodule" (e.g model.submodule.conv1 -> model.conv1)
                # and "module" (e.g features.module.conv2d -> features.conv2d)
                # they are created during DDP training, different from origin model
                for key in list(resume_point['state_dict'].keys()):
                    split_key = key.split('.')
                    split_origin = copy.deepcopy(split_key)
                    for item in split_origin:
                        if item == "module":
                            split_key.remove("module")
                    resume_point_replace[".".join(split_key)] = resume_point['state_dict'][key]
            else:
                resume_point_replace = resume_point['state_dict']
            model.load_state_dict(resume_point_replace, strict=True if args.device=='gpu' else False)
        else:
            print("ERROR: Fail to load Resume checkpoint from {}, file not exist".format(args.resume))
            return

        begin = time.time()

        mlperf_log.ncf_print(key=mlperf_log.EVAL_START, value=args.start_epoch)

        hr = float('nan')
        ndcg = float('nan')

        valid_results_file = os.path.join(run_dir, 'valid_results.csv')
        hr, ndcg = val_epoch(model, args, num_user=nb_users, output=valid_results_file, epoch=args.start_epoch)

        val_time = time.time() - begin
        print('Epoch {epoch}: HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f},'
                ' val_time = {val_time:.2f}'
            .format(epoch=args.start_epoch, K=args.topk, hit_rate=hr,
                    ndcg=ndcg, val_time=val_time))
        metric_collector = MetricCollector(enable_only_avglog=True)
        metric_collector.insert_metrics(net = "DLRM",
                                        accuracy = hr)
        metric_collector.dump()

        mlperf_log.ncf_print(key=mlperf_log.EVAL_ACCURACY, value={"epoch": args.start_epoch, "value": hr})
        mlperf_log.ncf_print(key=mlperf_log.EVAL_TARGET, value=args.threshold)
        mlperf_log.ncf_print(key=mlperf_log.EVAL_STOP, value=args.start_epoch)

class AverageMeter(object):
    def __init__(self, name, fmt:':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        self.series = []

    def update(self, val, n=1):
        self.val = val
        self.series.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

if __name__ == '__main__':
    main()
