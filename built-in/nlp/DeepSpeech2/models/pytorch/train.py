import argparse
import errno
import json
import os
import random
import time

import sys

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss
from torch.nn.parallel import DistributedDataParallel
# from trains.train_factory import train_factory

import torch.nn.functional as F
import torch.distributed as dist

### Import Data Utils ###
sys.path.append('../')

from data.bucketing_sampler import BucketingSampler, SpectrogramDatasetWithLength
from data.data_loader import AudioDataLoader, SpectrogramDataset
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns
try:
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
except ImportError:
    print("import torch_mlu failed!")

import params

from eval_model import  eval_model

## Import tools/utils
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../tools/utils/")
from metric import MetricCollector

###########################################################
# Comand line arguments, handled by params except seed    #
###########################################################
parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')

parser.add_argument('--seed', default=0xdeadbeef, type=int, help='Random Seed')

parser.add_argument('--acc', default=23.0, type=float, help='Target WER')

parser.add_argument('--device', default='mlu', type=str, help='use mlu, gpu or cpu')

parser.add_argument('--distributed', default=False, type=bool, help='Training with a multiple process.')

parser.add_argument('--num_workers', default=1, type=int)

parser.add_argument('--start_epoch', default=-1, type=int)

parser.add_argument('--world_size', default=1, type=int)

parser.add_argument('--local_rank', default=1, type=int)
 
parser.add_argument('--iters', default=-1, type=int)

parser.add_argument('--eval_iters', default=-1, type=int)

def reduce_tensor(tensor, num_device):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if rt.is_floating_point():
        rt = rt/num_device
    else:
        rt = rt//num_device
    return rt

def to_np(x):
    return x.data.cpu().numpy()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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

class CTCLOSS(nn.Module):
    def __init__(self):
        super(CTCLOSS, self).__init__()
        self.loss = nn.CTCLoss()
    def forward(self, out, targets, sizes, target_sizes):
        device = out.device
        self = self.cpu()
        out = out.cpu()
        tmp = self.loss(out, targets, sizes, target_sizes)
        tmp = tmp.to(device)
        #tmp = tmp.to('mlu', non_blocking = True)
        return tmp

class WARP_CTCLOSS(nn.Module):
    def __init__(self):
        super(WARP_CTCLOSS, self).__init__()
        self.loss = CTCLoss()
    def forward(self, out, targets, sizes, target_sizes):
        device = out.device
        self = self.cpu()
        out = out.cpu()
        tmp = self.loss(out, targets, sizes, target_sizes)
        tmp = tmp.to(device)
        return tmp


def main():
    args = parser.parse_args()
    device = 'cuda' if args.device == 'gpu' else args.device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    TEST_ITERS=10

    if params.rnn_type == 'gru' and params.rnn_act_type != 'tanh':
      print("ERROR: GRU does not currently support activations other than tanh")
      sys.exit()

    if params.rnn_type == 'rnn' and params.rnn_act_type != 'relu':
      print("ERROR: We should be using ReLU RNNs")
      sys.exit()

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ["WORLD_SIZE"])

    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    if args.distributed:
      if args.device == "mlu":
        ct.set_device(args.local_rank)
      else:
        torch.cuda.set_device(args.local_rank)
      torch.distributed.init_process_group(backend='cncl' if args.device == 'mlu' else 'nccl', init_method='env://')
      #args.world_size = torch.distributed.get_world_size()
      args.local_rank = torch.distributed.get_rank()
      print('Training in distributed mode with multiple processes, 1 GPU or MLU per process. Process %d, total %d.'
            % (args.local_rank, args.world_size))
    else:
      print('Training with a single process.')

    save_folder = args.save_folder

    loss_results, cer_results, wer_results = torch.Tensor(params.epochs), torch.Tensor(params.epochs), torch.Tensor(params.epochs)
    best_wer = None
    try:
        os.makedirs(save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise
    criterion = WARP_CTCLOSS()

    with open(params.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))
    audio_conf = dict(sample_rate=params.sample_rate,
                      window_size=params.window_size,
                      window_stride=params.window_stride,
                      window=params.window,
                      noise_dir=params.noise_dir,
                      noise_prob=params.noise_prob,
                      noise_levels=(params.noise_min, params.noise_max))

    # init dataloader
    train_sampler = None
    shuffle = True
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=params.train_manifest, labels=labels,
                                       normalize=True, augment=params.augment)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=args.seed)
        shuffle = False
    train_loader = AudioDataLoader(train_dataset,
                                   batch_size=params.batch_size,
                                   shuffle=shuffle,
                                   num_workers=args.num_workers,
                                   pin_memory=True,
                                   drop_last=True,
                                   sampler=train_sampler)

    test_sampler = None
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=params.val_manifest, labels=labels,
                                      normalize=True, augment=False)
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = AudioDataLoader(test_dataset,
                                  batch_size=params.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  sampler=test_sampler)

    rnn_type = params.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"

    model = DeepSpeech(rnn_hidden_size = params.hidden_size,
                       nb_layers       = params.hidden_layers,
                       labels          = labels,
                       rnn_type        = supported_rnns[rnn_type],
                       audio_conf      = audio_conf,
                       bidirectional   = False,
                       rnn_activation  = params.rnn_act_type,
                       bias            = params.bias)

    if args.device == 'gpu':
        model         = model.cuda()
    elif args.device == 'mlu':
        model = model.to('mlu')

    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=params.lr,
                                momentum=params.momentum, nesterov=True,
                                weight_decay = params.l2)
    decoder = GreedyDecoder(labels)

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    if args.continue_from:
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from, map_location=torch.device('cpu'))
        new_package_state_dict = {}
        for key in package["state_dict"].keys():
            if args.distributed:
                new_key = "module." + key
            else:
                new_key = key
            new_package_state_dict[new_key] = package["state_dict"][key]
        model.load_state_dict(new_package_state_dict)
        optimizer.load_state_dict(package['optim_dict'])
        start_epoch = int(package.get('epoch', 1)) - 1  # Python index start at 0 for training
        start_iter = package.get('iteration', None)
        if start_iter is None:
            start_epoch += 1  # Assume that we saved a model after an epoch finished, so start at the next epoch.
            start_iter = 0
        else:
            start_iter += 1
        avg_loss = int(package.get('avg_loss', 0))

        if args.start_epoch != -1:
          start_epoch = args.start_epoch

        loss_results[:start_epoch], cer_results[:start_epoch], wer_results[:start_epoch] = package['loss_results'][:start_epoch], package[ 'cer_results'][:start_epoch], package['wer_results'][:start_epoch]
        print(loss_results)
        epoch = start_epoch

    else:
        avg_loss = 0
        start_epoch = 0
        start_iter = 0
        avg_training_loss = 0


    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ctc_time = AverageMeter()

    ## BENCHMARK_LOG and AVG_LOG test
    enable_only_benchmark = True if "BENCHMARK_LOG" in os.environ else False
    enable_only_avglog = True if "AVG_LOG" in os.environ else False
    metric_collector = MetricCollector(enable_only_benchmark=enable_only_benchmark,
                                       enable_only_avglog=enable_only_avglog,
                                       record_elapsed_time=True,
                                       record_hardware_time=True if args.device == 'mlu' else False)

    reduce_losses = None
    break_flag = False
    metric_collector.place()

    for epoch in range(start_epoch, params.epochs):
        model.train()
        end = time.time()
        for i, (data) in enumerate(train_loader, start=start_iter):
            if (i==args.iters):
                break_flag = True if (enable_only_benchmark and not enable_only_avglog) else False
                break
            if i == len(train_loader):
                break
            # if i == (len(train_loader) - args.iter_num):
            #     torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration = i, loss_results=loss_results,
            #                                     wer_results=wer_results, cer_results=cer_results), file_path)
            inputs, targets, input_percentages, target_sizes = data
            # measure data loading time
            data_time.update(time.time() - end)

            if args.device == 'gpu':
                inputs = inputs.cuda()
            elif args.device == 'mlu':
                inputs = inputs.to('mlu', non_blocking = True)

            out = model(inputs)
            out = out.transpose(0, 1)  # TxNxH


            seq_length = out.size(0)
            sizes = input_percentages.mul_(int(seq_length)).int()

            ctc_start_time = time.time()
            loss = criterion(out, targets, sizes, target_sizes)
            ctc_time.update(time.time() - ctc_start_time)

            loss = loss / inputs.size(0)  # average the loss by minibatch

            loss_sum = loss.data.sum()
            inf = float("inf")
            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.data.item()

            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), params.max_norm)
            # SGD step
            optimizer.step()
            if args.device == 'gpu':
                torch.cuda.synchronize()
            metric_collector.record()
            metric_collector.place()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'CTC Time {ctc_time.val:.3f} ({ctc_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                (epoch + 1), (i + 1), len(train_loader), batch_time=batch_time,
                data_time=data_time, ctc_time=ctc_time, loss=losses))
            if args.distributed:
                reduce_losses = reduce_tensor(loss, args.world_size)
            else:
                reduce_losses = losses.val

            del loss
            del out

        if break_flag:
            break
        avg_loss /= len(train_loader)

        print('Training Summary Epoch: [{0}]\t'
            'Average Loss {loss:.3f}\t'
            .format( epoch + 1, loss=avg_loss, ))

        start_iter = 0  # Reset start iteration for next epoch
        total_cer, total_wer = 0, 0
        model.eval()

        with torch.no_grad():
            test_iters = args.eval_iters if "AVG_LOG" not in os.environ else TEST_ITERS
            wer, cer = eval_model(model, test_loader, decoder, args, test_iters)

        loss_results[epoch] = avg_loss
        wer_results[epoch] = wer
        cer_results[epoch] = cer
        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(
            epoch + 1, wer=wer, cer=cer))

        save_model = model
        if args.distributed:
            save_model = model.module
        if args.checkpoint and (args.distributed == False or args.local_rank == 0):
            file_path = '%s/deepspeech_%d.pth.tar' % (save_folder, epoch + 1)
            torch.save(DeepSpeech.serialize(save_model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                            wer_results=wer_results, cer_results=cer_results),
                                            file_path)
        # anneal lr
        optim_state = optimizer.state_dict()
        optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / params.learning_anneal
        optimizer.load_state_dict(optim_state)
        print('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

        if (best_wer is None or best_wer > wer) and (args.distributed == False or args.local_rank == 0):
            print("Found better validated model, saving to %s" % args.model_path)
            torch.save(DeepSpeech.serialize(save_model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                            wer_results=wer_results, cer_results=cer_results)
                       , args.model_path)
            best_wer = wer

        avg_loss = 0

        #If set to exit at a given accuracy, exit. Stop Single-Card
        if params.exit_at_acc and (best_wer is not None and best_wer <= args.acc) and (args.distributed == False):
            break
        # Stop DDP Not Support

    metric_collector.insert_metrics(net="DeepSpeech2",
                                    batch_size=params.batch_size,
                                    precision="fp32",
                                    cards=args.world_size,
                                    DPF_mode = "ddp " if args.world_size > 1 else "single",
                                    accuracy = reduce_losses.data.item() if args.distributed else reduce_losses)
    if ((args.local_rank == 0) or not (args.world_size > 1)):
        metric_collector.dump()
    print("=======================================================")
    print("***Best WER = ", best_wer)
    for arg in vars(args):
      print("***%s = %s " %  (arg.ljust(25), getattr(args, arg)))
    print("=======================================================")

if __name__ == '__main__':
    main()
