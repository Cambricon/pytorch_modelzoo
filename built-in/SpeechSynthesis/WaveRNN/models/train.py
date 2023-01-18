import re
import os
import time
import copy
import random
import numpy as np
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
import torch
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from utils.dsp import save_wav
from utils.display import stream, simple_table
from utils.dataset_deepmind import get_deepmind_datasets
from utils import hparams as hp
from deepmind_version import WaveRNN
from utils.paths import Paths
import argparse
from utils import data_parallel_workaround
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../tools/utils/")

from metric import MetricCollector
try:
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
except ImportError:
    print("import torch_mlu failed!")

def parser_args(parser):
    parser.add_argument('-o', '--output', type=str, default='./output/', help='Directory to save checkpoints')
    parser.add_argument('--log-file', type=str, default='nvlog.json',help='Filename for logging')
    parser.add_argument('--seed', type=int, help='manually set random seed for torch')
    parser.add_argument('--device', default='MLU', type=str, help='set the type of hardware used for training.')
    parser.add_argument('-m', '--model-name', type=str, default='WaveRNN', help='Model to train')

    training = parser.add_argument_group('training setup')
    training.add_argument('--do-train', action='store_true', help='training network')
    training.add_argument('--num-workers',  default=8, type=int, help='number workers for dataloader')
    training.add_argument('--epochs',  default=10, type=int, help='Number of total epochs to run')
    training.add_argument('--iterations', default=1000, type=int, help='Number of total epochs to run')
    training.add_argument('--eval', default=0, type=int, help='Number of eval to run')
    training.add_argument('--seq-len', type=int, help='Seq len to train')
    training.add_argument('--generate', action='store_true', help='Whether generate .wav file')
    training.add_argument('--amp', action='store_true', help='use pytorch amp for mixed precision training')
    training.add_argument('--checkpoint-path', type=str, default='',help='Checkpoint path to resume training')
    training.add_argument('--cudnn-enabled', action='store_true', help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', action='store_true', help='Run cudnn benchmark')
    training.add_argument('--cudnn-deterministic', action='store_true', help='Run cudnn benchmark')
    training.add_argument('--num-per-checkpoint', type=int, default=-1, help='Number of epochs per checkpoint')
    training.add_argument('--resume-from-last', action='store_true', help='Resumes training from the last checkpoint')
    training.add_argument('--resume-multi-device', action='store_true', help='Resumes training from the last multidevice checkpoint.')
    training.add_argument('--hp-file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    training.add_argument('--sample_rate', default=22050, type=int, help='The sample_rate to generate .wav file')

    optimization = parser.add_argument_group('optimization setup')
    optimization.add_argument('--lr', '-l', type=float,  help='[float] override hparams.py learning rate')
    optimization.add_argument('--batch-size', type=int, help='[int] override hparams.py batch size')

    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int, help='Rank of the process, do not set! Done by multiproc module')
    distributed.add_argument('--world-size', default=1, type=int, help='Number of processes, do not set! Done by multiproc module')
    distributed.add_argument('--dist-url', type=int, default=23456, help='Url used to set up distributed training')
    distributed.add_argument('--group-name', type=str, default='group_name', required=False, help='Distributed group name')
    distributed.add_argument('--dist-backend', default='nccl', type=str, choices={'nccl', 'cncl'}, help='Distributed run backend')

    benchmark = parser.add_argument_group('benchmark')
    benchmark.add_argument('--bench-class', type=str, default='')
    return parser

def init_distributed(args, world_size, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=world_size, rank=rank, group_name=group_name)
    print("Done initializing distributed")

def init_mlu_distributed(args, world_size, rank, group_name):
    assert torch.is_mlu_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")
    import torch_mlu.core.mlu_model as ct
    ct.set_device(rank % ct.device_count())
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=world_size, rank=rank, group_name=group_name)
    print('Done initializing distributed')

def get_last_checkpoint_filename(output_dir, model_name):
    symlink = os.path.join(output_dir, "checkpoint_{}_last.pt".format(model_name))
    if os.path.exists(symlink):
        print("Loading checkpoint from symlink", symlink)
        return os.path.join(output_dir, os.readlink(symlink))
    else:
        print("No last checkpoint available - starting from epoch 0 ")
        return ""


def load_checkpoint(model, optimizer, epoch, args, local_rank, scaler):
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')

    epoch[0] = checkpoint['epoch']+1
    device_id = 0
    if args.device == 'MLU':
        device_id = local_rank % ct.device_count()
    else:
        device_id = local_rank % torch.cuda.device_count()
    if 'random_rng_states_all' in checkpoint:
        try:
            torch.random.set_rng_state(checkpoint['random_rng_states_all'][device_id])
        except:
            torch.random.set_rng_state(checkpoint['random_rng_states_all'])
    elif 'random_rng_state' in checkpoint:
        torch.random.set_rng_state(checkpoint['random_rng_state'])
    else:
        raise Exception("Model checkpoint must have either 'random_rng_state' or 'random_rng_states_all' key.")
    resume_point_replace = {}
    if args.resume_multi_device:
        for key in checkpoint['state_dict'].keys():
            split_key = key.split('.')
            split_origin = copy.deepcopy(split_key)
            for item in split_origin:
                if item == "module":
                    split_key.remove("module")
            resume_point_replace[".".join(split_key)] = checkpoint['state_dict'][key]
    else:
        resume_point_replace = checkpoint['state_dict']
    model.load_state_dict(resume_point_replace, strict=True if args.device != 'MLU' else False)
    optimizer.load_state_dict(checkpoint['optimizer'])

    if args.amp and 'amp' in checkpoint:
        scaler.load_state_dict(checkpoint['amp'])

def save_checkpoint(model, optimizer, epoch, model_name, local_rank, world_size, args, scaler):
    if args.device == 'MLU':
        random_rng_state = torch.random.get_rng_state().to('mlu')
        cuda_rng_state = torch.empty([]).to('mlu')
    else:
        random_rng_state = torch.random.get_rng_state().cuda()
        cuda_rng_state = torch.cuda.get_rng_state(local_rank).cuda()

    random_rng_states_all = [torch.empty_like(random_rng_state) for _ in range(world_size)]
    cuda_rng_states_all = [torch.empty_like(cuda_rng_state) for _ in range(world_size)]

    if world_size > 1:
        dist.all_gather(random_rng_states_all, random_rng_state)
        dist.all_gather(cuda_rng_states_all, cuda_rng_state)
    else:
        random_rng_states_all = [random_rng_state]
        cuda_rng_states_all = [cuda_rng_state]

    random_rng_states_all = torch.stack(random_rng_states_all).cpu()
    cuda_rng_states_all = torch.stack(cuda_rng_states_all).cpu()

    if local_rank == 0:
        checkpoint = {'epoch': epoch,
                      'cuda_rng_state_all': cuda_rng_states_all,
                      'random_rng_states_all': random_rng_states_all,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        if scaler:
            checkpoint['amp'] = scaler.state_dict()

        checkpoint_filename = "checkpoint_{}_{}.pt".format(model_name, epoch)
        checkpoint_path = os.path.join(args.output, checkpoint_filename)
        print("Saving model and optimizer state at epoch {} to {}".format(
            epoch, checkpoint_path))
        torch.save(checkpoint, checkpoint_path)

        symlink_src = checkpoint_filename
        symlink_dst = os.path.join(
            args.output, "checkpoint_{}_last.pt".format(model_name))
        if os.path.exists(symlink_dst) and os.path.islink(symlink_dst):
            print("Updating symlink", symlink_dst, "to point to", symlink_src)
            os.remove(symlink_dst)

        os.symlink(symlink_src, symlink_dst)


def main():
    parser = argparse.ArgumentParser(description='PyTorch WaveRNN training')
    parser = parser_args(parser)
    args, _ = parser.parse_known_args()

    hp.configure(args.hp_file)  # load hparams from file

    if args.device == 'MLU':
        args.dist_backend = 'cncl'

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(seed=args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        local_rank = args.rank
        world_size = args.world_size

    if not os.path.exists(args.output) and local_rank == 0:
        os.makedirs(args.output)

    distributed_run = world_size > 1

    if local_rank == 0:
        log_file = os.path.join(args.output, args.log_file)
        DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, log_file),
                                StdOutBackend(Verbosity.VERBOSE)])
    else:
        DLLogger.init(backends=[])

    for k,v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})
    DLLogger.log(step="PARAMETER", data={'model_name':'WaveRNN_PyT'})

    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = args.cudnn_deterministic

    if distributed_run:
        args.dist_url = 'tcp://localhost:'+str(args.dist_url)
        if args.device == "CPU":
            print("The CPU device platform does not support distributed operation.")
            return
        if args.device == 'MLU':
            init_mlu_distributed(args, world_size, local_rank, args.group_name)
        else:
            init_distributed(args, world_size, local_rank, args.group_name)

    if args.seq_len is None:
        args.seq_len = hp.dpm_seq_len
    if args.lr is None:
        args.lr = hp.dpm_lr
    if args.batch_size is None:
        args.batch_size = hp.dpm_batch_size
    if args.sample_rate is None:
        args.sample_rate = hp.sample_rate        

    paths = Paths(hp.data_path)
    batch_size = args.batch_size
    if args.device == "CPU":
        device = torch.device('cpu')
    if args.device == "GPU":
        device = torch.device('cuda')
        if batch_size % torch.cuda.device_count():
            raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    if args.device == "MLU":
        device = torch.device('mlu')
        if batch_size % ct.device_count():
            raise ValueError('`batch_size` must be evenly divisible by n_mlus!')

    print('Using device:', device)
    print('\nInitialising Model...\n')

    scaler = GradScaler() if args.amp else None

    train_set, test_set = get_deepmind_datasets(paths.data, batch_size, args.seq_len, distributed_run, args.num_workers)
    if len(test_set) < args.eval:
        print('number of eval({}) should less than testset({})'.format(args.eval,len(test_set)))
        return

    model = WaveRNN(hidden_size=hp.dpm_hidden_size, quantisation=hp.dpm_quantisation).to(device)
    optimizer = optim.Adam(model.parameters(),lr=args.lr)

    start_epoch = [1]
    if args.resume_from_last:
        args.checkpoint_path = get_last_checkpoint_filename(args.output, args.model_name)

    if args.checkpoint_path is not "":
        load_checkpoint(model, optimizer, start_epoch, args, local_rank, scaler)

    if distributed_run:
        model = DDP(model, device_ids=[local_rank],output_device=local_rank)

    start_epoch = start_epoch[0]
    simple_table([('Num Epoch', str(args.epochs-start_epoch+1)),
                  ('per_iters', str(args.iterations)),
                  ('Batch Size', batch_size),
                  ('LR', args.lr),
                  ('Sequence Len', args.seq_len)])
    if args.do_train:
        dpm_train_loop(model, optimizer, train_set, batch_size, start_epoch, distributed_run, local_rank, world_size, args, scaler)
        print('Training Complete.')

    if args.eval:
        print('eval now...')
        validate(model, optimizer, test_set ,batch_size, distributed_run, local_rank, args)

    if args.generate:
        print('generate now...')
        model_module =  model.module
        generate(model_module, args.sample_rate * 5, args.output)

def dpm_train_loop(model, optimizer, train_set, batch_size, start_epoch, distributed_run, local_rank, world_size, args, scaler):
    model.train()
    benchMark_flag = True if os.getenv('BENCHMARK_LOG') else False
    avg_log_flag = True if os.getenv('AVG_LOG') else False
    for e in range(start_epoch, args.epochs+1):
        model.train()
        if args.device == "GPU":
            torch.cuda.synchronize()
        if args.device == "MLU":
            ct.current_queue().synchronize()
        if distributed_run:
            train_set.sampler.set_epoch(e)
        train(e, args, train_set, model, optimizer, batch_size, distributed_run, local_rank, world_size, scaler, benchMark_flag, avg_log_flag)

def train(e, args, train_set, model, optimizer, batch_size, distributed_run, local_rank, world_size, scaler, benchMark_flag, avg_log_flag):
    running_loss = 0
    seq_len = args.seq_len
    metric_collector = MetricCollector(enable_only_benchmark=True, record_elapsed_time=True,
        record_hardware_time=True if args.device == 'MLU' else False)
    device_count = ct.device_count() if args.device=="MLU" else torch.cuda.device_count()
    metric_collector.place()
    for step, (coarse_classes, fine_classes) in enumerate(train_set, 1):
        iter_start_time = time.perf_counter()
        if step == args.iterations +1:
            print('The program iteration runs out. iterations:%d' % args.iterations)
            break
        if args.device == "MLU":
            coarse_classes = coarse_classes.to(ct.mlu_device(), non_blocking=True)
            fine_classes = fine_classes.to(ct.mlu_device(), non_blocking=True)
        if args.device == "GPU":
            coarse_classes = coarse_classes.cuda()
            fine_classes = fine_classes.cuda()

        if distributed_run:
            hidden = Variable(model.module.get_initial_hidden(batch_size))
        else:
            hidden = Variable(model.get_initial_hidden(batch_size))
        rand_idx = np.random.randint(0, coarse_classes.shape[1] - seq_len - 1)
        x_coarse = coarse_classes[:, rand_idx:rand_idx + seq_len]
        x_coarse = x_coarse / 127.5 -1.
        x_fine = fine_classes[:, rand_idx:rand_idx + seq_len]
        x_fine = x_fine /127.5 -1.

        y_coarse = coarse_classes[:, rand_idx + 1:rand_idx + seq_len + 1]
        y_fine = fine_classes[:, rand_idx + 1: rand_idx + seq_len + 1]

        loss = 0

        for i in range(seq_len) :
            x_c_in = x_coarse[:, i:i + 1]
            x_f_in = x_fine[:, i:i + 1]
            x_input = torch.cat([x_c_in, x_f_in], axis=1)
            x_input = Variable(x_input)

            c_target = y_coarse[:, i]
            f_target = y_fine[:, i]
            c_target = Variable(c_target)
            f_target = Variable(f_target)

            current_coarse = c_target.float() / 127.5 - 1.
            current_coarse = current_coarse.unsqueeze(-1)

            with torch.cuda.amp.autocast(enabled=args.amp):
                out_coarse, out_fine, hidden = model(x_input, hidden, current_coarse)
                c_target = c_target.long()
                f_target = f_target.long()
                loss_coarse = F.cross_entropy(out_coarse, c_target)
                loss_fine = F.cross_entropy(out_fine, f_target)
                loss += (loss_coarse + loss_fine)


        running_loss += (loss.item() / seq_len)
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.device == "GPU":
            torch.cuda.synchronize()
        if args.device == "MLU":
            ct.current_queue().synchronize()
        metric_collector.record()
        metric_collector.place()
        iter_stop_time = time.perf_counter()
        iter_time = iter_stop_time - iter_start_time
        DLLogger.log(step=(e,step), data={'train_loss': round(running_loss/step, 6), 'time' : iter_time})
        metric_collector.insert_metrics(
            net = args.model_name,
            batch_size = args.batch_size,
            precision = "amp" if args.amp else "fp32",
            cards = device_count if local_rank == 0 else 1,
            DPF_mode = "ddp " if distributed_run else "single")

    if (not distributed_run) or local_rank == 0:
        metric_collector.dump()

    if args.num_per_checkpoint >0 and e % args.num_per_checkpoint == 0:
        save_checkpoint(model, optimizer, e, args.model_name, local_rank, world_size, args, scaler)

def generate(model, seq_len, output_dir):
    output, c, f = model.generate(seq_len)
    wav_file_path = os.path.join(output_dir, 'gen_wav')
    if not os.path.exists(wav_file_path):
        os.makedirs(wav_file_path)
    save_wav(output, '{}/{}.wav'.format(wav_file_path,seq_len))
    print('.wav file generated successfully')


def validate(model, optimizer, test_set ,batch_size, distributed_run, local_rank, args):
    seq_len = args.seq_len
    model.eval()
    if args.device == "GPU":
        torch.cuda.synchronize()
    if args.device == "MLU":
        ct.current_queue().synchronize()
    running_loss = 0
    for step, (coarse_classes, fine_classes) in enumerate(test_set, 1):
        if step == args.eval+1:
            print('\nThe program eval runs out. num:%d' % args.eval)
            break
        if args.device == "MLU":
            coarse_classes = coarse_classes.to(ct.mlu_device(), non_blocking=True)
            fine_classes = fine_classes.to(ct.mlu_device(), non_blocking=True)
        if args.device == "GPU":
            coarse_classes = coarse_classes.cuda()
            fine_classes = fine_classes.cuda()

        if distributed_run:
            hidden = Variable(model.module.get_initial_hidden(batch_size))
        else:
            hidden = Variable(model.get_initial_hidden(batch_size))
        coarse_classes = torch.reshape(coarse_classes,(coarse_classes.shape[1],coarse_classes.shape[2]))
        fine_classes = torch.reshape(fine_classes, (fine_classes.shape[1],fine_classes.shape[2]))
        rand_idx = np.random.randint(0, coarse_classes.shape[1] - seq_len - 1)
        x_coarse = coarse_classes[:, rand_idx:rand_idx + seq_len]
        x_coarse = x_coarse / 127.5 - 1.
        x_fine = fine_classes[:, rand_idx:rand_idx + seq_len]
        x_fine = x_fine / 127.5 - 1.

        y_coarse = coarse_classes[:, rand_idx + 1:rand_idx + seq_len + 1]
        y_fine = fine_classes[:, rand_idx + 1: rand_idx + seq_len + 1]
        loss = 0
        for i in range(seq_len) :
            x_c_in = x_coarse[:, i:i + 1]
            x_f_in = x_fine[:, i:i + 1]
            x_input = torch.cat([x_c_in, x_f_in], axis=1)
            x_input = Variable(x_input)

            c_target = y_coarse[:, i]
            f_target = y_fine[:, i]
            current_coarse = c_target.float() / 127.5 - 1.
            current_coarse = current_coarse.unsqueeze(-1)

            out_coarse, out_fine, hidden = model(x_input, hidden, current_coarse)
            c_target = c_target.long()
            f_target = f_target.long()
            loss_coarse = F.cross_entropy(out_coarse, c_target)
            loss_fine = F.cross_entropy(out_fine, f_target)
            loss += (loss_coarse + loss_fine)

        running_loss += (loss.item() / seq_len)
        if local_rank==0:
            stream('Step: {}/{} --- Loss: {:.3f}'.format(step, args.eval, running_loss/step))
    if (not distributed_run) or local_rank == 0:
        metric_collector = MetricCollector(enable_only_avglog=True)
        metric_collector.insert_metrics(net = args.model_name,
                                        accuracy = [running_loss/args.eval])
        metric_collector.dump()

if __name__ == "__main__":
    main()
