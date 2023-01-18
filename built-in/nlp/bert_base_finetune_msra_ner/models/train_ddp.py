"""Train and evaluate the model"""

import argparse
import logging
import numpy as np
import random
import os
import re

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

import utils


from pytorch_pretrained_bert import BertForTokenClassification
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP


from data_loader_ddp import TrainDataSet, ValDataSet, TestDataSet
from evaluate import evaluate
from metrics import f1_score
from metrics import classification_report
import time
import sys
from torch.cuda.amp import autocast, GradScaler
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../tools/utils/")
from metric import MetricCollector


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/msra', help="Directory containing the dataset")
parser.add_argument('--bert_model_dir', default='bert-base-chinese-pytorch', help="Directory containing the BERT model in PyTorch")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--seed', type=int, default=2019, help="random seed for initialization")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")
parser.add_argument('--gpu', default=None, type=int, help="Whether to use single GPUs if available")
parser.add_argument('--fp16', default=False, action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers.')
parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training.')
parser.add_argument('--rank', default=-1, type=int, help='node rank fro distributed training.')
parser.add_argument('--dist-url', default="tcp://127.0.0.1:8213", type=str, help='url used to set up distributed training.')
parser.add_argument('--dist-backend', default="nccl", type=str, help='distributed backend.')
parser.add_argument('--distributed', default=False, action='store_true', help='Use ddp to train.')
parser.add_argument('--device', default='cpu', type=str, help='Use cpu gpu or mlu device')
parser.add_argument('--device_id', default=None, type=int,
                        help='Use specified device for training, useless in multiprocessing distributed training')
parser.add_argument('--iters', type=int, default=-1, metavar='N',
                        help='train iters per epoch')
parser.add_argument('--eval_iters', type=int, default=-1, metavar='N',
                        help='eval iters per epoch')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--nproc_per_node", type=int, default=1,
                    help="The number of processes to launch on each node, "
                          "for GPU training, this is recommended to be set "
                          "to the number of GPUs in your system so that "
                          "each process can be bound to a single GPU.")
parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
parser.add_argument('--run_epochs', type=int, default=-1,
                        help='epochs to train')
parser.add_argument('--batch_size', type=int, default=0,
                        help='training batch size. If set, it will override the parameters in json')
parser.add_argument('--max_bitwidth',
                    action='store_true',
                    help='use Max Bitwidth of MLU training')
parser.add_argument('--cnmix', action='store_true', default=False,
                    help='use cnmix for mixed precision training')
parser.add_argument('--opt_level', type=str, default='O1',
                        help='choose level of mixing precision')
parser.add_argument('--pyamp', action='store_true', default=False,
                    help='use pytorch amp for mixed precision training')

args = parser.parse_args()
if args.device == 'mlu':
    import torch_mlu
    import torch_mlu.core.mlu_model as ct

if args.cnmix:
    import cnmix

scaler = None
if args.pyamp:
    scaler = GradScaler()

def train(model, train_loader, optimizer, scheduler, params, args):
    """Train the model on `steps` batches"""
    # set model to training mode
    scheduler.step()
    model.train()
    adaptive_cnt = int(os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT')) if (
            os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT') is not None
            ) else 0
    # a running average object for loss
    loss_avg = utils.RunningAverage()
    batch_avg = utils.RunningAverage()
    batch_time_benchmark = []
    end = time.time()

    # for internal benchmark test
    metric_collector = MetricCollector(
        enable_only_benchmark=True,
        record_elapsed_time=True,
        record_hardware_time=True if args.device == 'mlu' else False)
    metric_collector.place()

    t = enumerate(train_loader)
    for i, (batch_data, batch_tags) in t:
        if i == args.iters:
            break
        # fetch the next training batch
        if args.device == 'mlu':
            batch_data = batch_data.to(ct.mlu_device(), non_blocking=True)
            batch_tags = batch_tags.to(ct.mlu_device(), non_blocking=True)
        else:
            batch_data = batch_data.to(params.device)
            batch_tags = batch_tags.to(params.device)
        batch_masks = batch_data.gt(0)

        # compute model output and loss
        with autocast(enabled=args.pyamp):
            loss = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)

        if params.n_device > 1:
            loss = loss.mean()  # mean() to average on multi-gpu

        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        #optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss)
        elif args.cnmix:
            with cnmix.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        elif args.pyamp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip_grad)

        # performs updates using calculated gradients
        if args.pyamp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # update the average loss
        loss_avg.update(loss.item())
        batch_avg.update(time.time() - end)
        if os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT') is not None and i > adaptive_cnt:
            batch_time_benchmark.append(time.time() - end)
        end = time.time()

        # MetricCollector record
        metric_collector.record()
        metric_collector.place()
        
        if args.rank==0:
            logging.info("rank={},train:iter={}/{},loss={}, loss_avg={}".format(args.rank, i, len(train_loader), loss.item(), loss_avg()))

    if args.cnmix:
        precision = args.opt_level
    elif args.pyamp:
        precision = "amp"
    else:
        precision = "fp32"
    metric_collector.insert_metrics(
        net = "bert_base_finetune_msra_ner",
        batch_size = args.batch_size,
        precision = precision,
        cards = args.nproc_per_node,
        DPF_mode = "ddp ")
    if (args.local_rank == 0):
        metric_collector.dump()


def evaluate(model, val_loader, params, mark='Eval', verbose=False):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    idx2tag = params.idx2tag

    true_tags = []
    pred_tags = []

    # a running average object for loss
    loss_avg = utils.RunningAverage()
    t = enumerate(val_loader)
    for i, (batch_data, batch_tags) in t:
        if i == args.eval_iters:
            break
        # fetch the next evaluation batch
        if args.device == 'mlu':
            batch_data = batch_data.to(ct.mlu_device(), non_blocking=True)
            batch_tags = batch_tags.to(ct.mlu_device(), non_blocking=True)
        else:
            batch_data.to(params.device)
            batch_tags.to(params.device)
        batch_masks = batch_data.gt(0)

        loss = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)
        if params.n_device > 1:
            loss = loss.mean()
        loss_avg.update(loss.item())
        if args.rank==0:
            logging.info("rank={},val:iter={}/{},loss={}, loss_avg={}".format(args.rank, i, len(val_loader), loss.item(), loss_avg()))

        batch_output = model(batch_data, token_type_ids=None, attention_mask=batch_masks)  # shape: (batch_size, max_len, num_labels)

        batch_output = batch_output.detach().cpu().numpy()
        batch_tags = batch_tags.to('cpu').numpy()

        pred_tags.extend([idx2tag.get(idx) for indices in np.argmax(batch_output, axis=2) for idx in indices])
        true_tags.extend([idx2tag.get(idx) for indices in batch_tags for idx in indices])
    assert len(pred_tags) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    f1 = f1_score(true_tags, pred_tags)
    metrics['loss'] = loss_avg()
    metrics['f1'] = f1
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mark) + metrics_str)
    logging.info(metrics_str)
    if (args.local_rank == 0):
        metric_collector = MetricCollector(enable_only_avglog=True)
        metric_collector.insert_metrics(net = "bert_base_finetune_msra_ner",
                                    accuracy = [f1])
        metric_collector.dump()

    if verbose:
        report = classification_report(true_tags, pred_tags)
        logging.info(report)
    return metrics




def main():
    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    # Refine cuda and cpu config
    use_cuda = False
    if args.device == 'gpu':
        use_cuda = True if torch.cuda.is_available() else False
    elif args.device == 'mlu':
        use_mlu = True

    args.distributed = args.multiprocessing_distributed or int(os.environ["WORLD_SIZE"]) > 1

    ndevs_per_node = 1
    if use_cuda:
        ndevs_per_node = torch.cuda.device_count()
        args.distributed = False if args.gpu else args.distributed
        params.device = torch.device('cuda')
    elif use_mlu:
        # ndevs_per_node = ct.device_count()
        ndevs_per_node = ct.device_count() if args.multiprocessing_distributed else args.nproc_per_node
        params.device = torch.device('mlu')
    else:
        print('Only detect cpu, args.gpu and args.distributed not working.')
        args.gpu = None
        args.distributed = None
        params.device = torch.device('cpu')
    # Set the random seed for reproducible experiments
    if args.seed is not None:
        params.seed = args.seed
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed_all(args.seed)  # set random seed for all GPUs
            cudnn.deterministic = True
    # Prepare for distributed training
    params.n_device = ndevs_per_node
    logging.info("device: {}, n_device: {}, 16-bits training: {}".format(params.device, params.n_device, args.fp16))
    if args.dist_url == "env://" or os.getenv('WORLD_SIZE'):
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:
        args.world_size = ndevs_per_node * args.world_size
    if args.multiprocessing_distributed:
        mp.spawn(main_worker, nprocs=ndevs_per_node, args=(ndevs_per_node, args, params))
    else:
        main_worker(args.local_rank, args.nproc_per_node, args, params)

def main_worker(dev_id, ndevs_per_node, args, params):
    args.device_id = dev_id

    if args.device_id is not None:
        print("Use device_id: {} for training".format(args.device_id))

    if args.distributed:
        if args.dist_url == "env://" or os.getenv('WORLD_SIZE'):
            args.rank = int(os.environ["RANK"])
        else:
            args.rank = args.rank * ndevs_per_node + dev_id
        if args.device == 'mlu' or args.device == 'gpu':
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
            if os.getenv('BENCHMARK_LOG') is None:
                batch_size = int(params.batch_size / args.world_size)
            else:
                batch_size = int(args.batch_size)
    else:
        batch_size = params.batch_size

    # Prepare train and valid dataloader
    train_data = TrainDataSet(args.data_dir, args.bert_model_dir, params, token_pad_idx=0)
    val_data = ValDataSet(args.data_dir, args.bert_model_dir, params, token_pad_idx=0)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False)

    train_val_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False)
    # Prepare model
    model = BertForTokenClassification.from_pretrained(args.bert_model_dir, num_labels=len(params.tag2idx), len_train=len(train_data), args=args)
    if args.fp16:
        model.half()
    if args.device_id is not None:
        if args.device == 'gpu':
            torch.cuda.set_device(args.device_id)
            model.cuda(args.device_id)
        else:
            ct.set_device(args.device_id)
            model.to(ct.mlu_device())
    else:
        if args.device == 'gpu':
            model.cuda()
        else:
            model.mlu()


    # Prepare optimizer
    if params.full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("lease install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=params.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch))
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = optim.Adam(optimizer_grouped_parameters, lr=params.learning_rate)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch))

    if args.cnmix:
        model. optimizer = cnmix.initialize(model, optimizer, opt_level=args.opt_level)
        cnmix.core.cnmix_set_amp_quantify_params('all',{'batch_size': params.batch_size,
                                                        'data_num': len(train_loader) * params.batch_size,
                                                        "quantify_rate": 1})

    if args.distributed:
        if args.device_id is not None:
            args.workers = int((args.workers + ndevs_per_node - 1) / ndevs_per_node)
            model = DDP(model, device_ids=[args.device_id], broadcast_buffers=False, find_unused_parameters=True)
        else:
            model = DDP(model, broadcast_buffers=False, find_unused_parameters=True)

    if args.restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, scaler, optimizer)

    # Train and evaluate the model
    best_val_f1 = 0.0
    patience_counter = 0

    if args.run_epochs != -1:
        run_epochs = args.run_epochs
    else:
        run_epochs = params.epoch_num
    for epoch in range(1, run_epochs + 1):
        if args.distributed:
             train_sampler.set_epoch(epoch)
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, run_epochs))
        # Train for one epoch on training set
        train(model, train_loader, optimizer, scheduler, params, args)
        if os.getenv('BENCHMARK_LOG'):
            break

        # Evaluate for one epoch on training set and validation set
        #train_metrics = evaluate(model, train_val_loader, params, mark='Train')
        val_metrics = evaluate(model, val_loader, params, mark='Val')

        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1

        # Save weights of the network
        if args.rank == 0:
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            optimizer_to_save = optimizer.optimizer if args.fp16 else optimizer
            save_file_path = os.path.join("bert_msra" + str(epoch) + ".pth")
            checkpoint_state = {'epoch': epoch + 1, 'state_dict': model_to_save.state_dict(), 'optim_dict': optimizer_to_save.state_dict()}
            if args.cnmix:
                checkpoint_state["cnmix"] = cnmix.state_dict()
            if args.pyamp:
                checkpoint_state["amp"] = scaler.state_dict()
            utils.save_checkpoint(checkpoint_state,
                                  is_best=improve_f1>0,
                                  checkpoint=args.model_dir)
        if improve_f1 > 0:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping and logging best f1
        if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == run_epochs:
            logging.info("Best val f1: {:05.2f}".format(best_val_f1))
            break


if __name__ == '__main__':
    main()
