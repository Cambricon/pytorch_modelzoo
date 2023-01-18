from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
import re
from utils import *
from dataset import *
import copy
import time

import crnn as crnnmodel
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../tools/utils/")
from metric import MetricCollector

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

try:
    if torch.is_mlu_available():
        import torch_mlu
        import torch_mlu.core.mlu_model as ct
        import torch_mlu.core.mlu_quantize as qt
except:
    print("Cambricon CATCH is not available!")

parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', required=True, help='path to dataset')
parser.add_argument('--valRoot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--mlu', action='store_true', help='enables mlu')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--expr_dir', default='./checkpoint_8', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1, help='Interval to be saved')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--ddp', default=False, help='Multi-processing distributed training')
parser.add_argument('--iter', type=int, default=-1, help='the training iterations')
parser.add_argument('--cnmix', action='store_true', default=False, help='use cnmix for mixed precision training')
parser.add_argument('--opt_level', type=str, default='O1', help='choose level of mixing precision')
parser.add_argument('--cudnn_lstm', action='store_true', help='if GPU, using cudnn LSTM; if MLU, using cnnl LSTM; otherwise using multi-operators')
parser.add_argument('--start_epoch', type=int, default='0', help='first start epoch')

opt = parser.parse_args()

if opt.cnmix:
    import cnmix


def setup(rank, world_size, backend='nccl'):
    if os.getenv("MASTER_ADDR") is None:
        print("[Warning]: MASTER_ADDR is set to 'localhost' as default")
        os.environ['MASTER_ADDR'] = 'localhost'
    if os.getenv("MASTER_PORT") is None:
        print("[Warning]: MASTER_PORT is set to '57834' as default")
        os.environ['MASTER_PORT'] = '57834'

    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def main():
    print(opt)
    if not os.path.exists(opt.expr_dir):
        os.makedirs(opt.expr_dir)

    cudnn.benchmark = True
    cudnn.enabled = True


    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    try:
        if torch.is_mlu_available() and not opt.mlu:
            print("WARNING: You have a MLU device, so you should probably run with --mlu")
    except:
        print("Cambricon CATCH is not available!")

    if opt.ddp:
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker,
        args=(opt, ),
        nprocs=opt.ngpu,
        join=True)
    else:
        # Simply call main_worker function
        main_worker(0, opt)


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main_worker(rank, opt):
    opt.rank = rank
    batch_size = opt.batchSize
    if opt.ddp:
        print(f"Running DDP on rank {rank}.")
        if torch.cuda.is_available():
            torch.cuda.set_device(opt.rank)
            setup(rank, opt.ngpu)
        else:
            ct.set_device(opt.rank)
            setup(rank, opt.ngpu, backend = "cncl")
        opt.manualSeed = (opt.manualSeed + torch.distributed.get_rank()) % 2**32

    print("Using seed = {}".format(opt.manualSeed))
    torch.manual_seed(opt.manualSeed)
    np.random.seed(seed=opt.manualSeed)
    random.seed(opt.manualSeed)

    train_dataset = lmdbDataset(root=opt.trainRoot)
    assert train_dataset
    if not opt.random_sample:
        sampler = randomSequentialSampler(train_dataset, batch_size)
    else:
        sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=False, sampler=sampler,
        num_workers=int(opt.workers),
        collate_fn=alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio), pin_memory=True)
    test_dataset = lmdbDataset(
        root=opt.valRoot, transform=resizeNormalize((100, 32)))

    nclass = len(opt.alphabet) + 1
    nc = 1

    opt.converter = strLabelConverter(opt.alphabet)
    # criterion = CTCLoss()
    try:
        from warpctc_pytorch import CTCLoss
        criterion = CTCLoss()
    except:
        print("WARNING: warpctc is not supported !")
        from torch.nn import CTCLoss
        cudnn.enabled = False
        criterion = torch.nn.CTCLoss(reduction='sum', zero_infinity=True)

    if torch.backends.cudnn.is_available():
        cudnn.enabled = True
        print("cudnn is available!")

    if opt.cudnn_lstm:
        print("Using cudnn/cnnl LSTM!")
        cudnn.enabled = True
        cudnn.set_flags(True, False, False)
    else:
        cudnn.enabled = True
        cudnn.set_flags(False, False, False)

    crnn = crnnmodel.CRNN(opt.imgH, nc, nclass, opt.nh)
    # crnn  = torch.jit.script(crnn)
    crnn.apply(weights_init)

    has_resume_optim = False
    if opt.pretrained != '':
        print('loading pretrained model from %s' % opt.pretrained)
        pretrained_weights = torch.load(opt.pretrained, map_location=torch.device('cpu'))
        pretrained_weights_replace = {}
        if 'state_dict' in pretrained_weights:
            for key in pretrained_weights['state_dict'].keys():
                split_key = key.split('.')
                split_origin = copy.deepcopy(split_key)
                for item in split_origin:
                    if item == "module":
                        split_key.remove("module")
                    elif item == "submodule":
                        split_key.remove("submodule")
                pretrained_weights_replace[".".join(split_key)] = pretrained_weights['state_dict'][key]
            opt.start_epoch = pretrained_weights['epoch']
            resume_optimizer = pretrained_weights['optimizer']
            has_resume_optim = True
        else:
            pretrained_weights_replace = {
                k.replace("module.", ""): v for k, v in pretrained_weights.items()
            }
        crnn.load_state_dict(pretrained_weights_replace, strict=True if opt.cuda else False)
    if rank == 0:
        print(crnn)

    image = torch.FloatTensor(batch_size, 3, opt.imgH, opt.imgH)
    text = torch.IntTensor(batch_size * 5)
    length = torch.IntTensor(batch_size)

    if opt.cuda:
        if opt.ddp:
            crnn = crnn.to(rank, non_blocking=True)
            crnn = DDP(crnn, device_ids=[rank]).to(rank, non_blocking=True)
            image = image.to(rank, non_blocking=True)
            criterion = criterion.cuda()
        else:
            crnn.cuda()
            crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
            image = image.cuda()
            criterion = criterion.cuda()
    elif opt.mlu:
        if opt.ddp:
            ct.set_device(rank)
            image = image.to(ct.mlu_device(), non_blocking=True)
            crnn = crnn.to(ct.mlu_device(), non_blocking=True)
        else:
            crnn.to(ct.mlu_device())

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                            betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(crnn.parameters())
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

    if opt.pretrained != '' and has_resume_optim:
        optimizer.load_state_dict(resume_optimizer)

    if opt.mlu:
        if opt.cnmix:
            cnmix.core.cnmix_set_amp_use_online(True)
            crnn, optimizer = cnmix.initialize(crnn, optimizer, opt_level=opt.opt_level)
            if opt.pretrained != '':
                if isinstance(pretrained_weights_replace, dict) and\
                      'cnmix' in pretrained_weights_replace:
                    cnmix.load_state_dict(pretrained_weights_replace['cnmix'])
        if opt.ddp:
            crnn = DDP(crnn, device_ids=[rank], find_unused_parameters=True)
            criterion = criterion.to(ct.mlu_device())
        else:
            crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
            image = image.to(ct.mlu_device())
            criterion = criterion.to(ct.mlu_device())

    opt.image = Variable(image)
    opt.text = Variable(text)
    opt.length = Variable(length)

    # loss averager
    loss_avg = averager()
    # time averager
    time_avg = averager()

    if opt.mlu and opt.cnmix:
        cnmix.cnmix_set_amp_quantify_params('all', {'batch_size': opt.batchSize,
                                                     'data_num': opt.batchSize * len(train_loader)})

    train(crnn, train_loader, test_dataset, criterion, optimizer, loss_avg, time_avg, opt)

    if opt.ddp:
        cleanup()


# 非常影响训练效率，建议不要在训练时做验证
def val(net, dataset, criterion, opt, max_iter=100):
    print('Start val')
    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    if opt.ddp:
        opt_batch_size = int(opt.batchSize / opt.ngpu)  # 分割batch
    else:
        opt_batch_size = opt.batchSize
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt_batch_size, num_workers=int(opt.workers), pin_memory=True)
    val_iter = iter(data_loader)

    if opt.ddp:
        epoch_size = int(len(data_loader) / opt.ngpu)  # 分割batch
    else:
        epoch_size = len(data_loader)

    i = 0
    n_correct = 0
    loss_avg = averager()

    max_iter = min(max_iter, epoch_size)
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        # cpu_text_new = [s for s in cpu_texts]
        # for i in range(len(cpu_text_new)):
        #     cpu_text_new[i] = cpu_text_new[i].replace("'", "")
        #     cpu_text_new[i] = cpu_text_new[i].replace('b', '')
        batch_size = cpu_images.size(0)
        loadData(opt.image, cpu_images)
        t, l = opt.converter.encode(cpu_texts)
        loadData(opt.text, t)
        loadData(opt.length, l)

        preds = net(opt.image)
        preds_size = Variable(torch.IntTensor([preds.size(1)] * batch_size))
        preds = preds.permute(1, 0, 2)  # to use CTCloss format ?
        cost = criterion(preds, opt.text, preds_size, opt.length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        # preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = opt.converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = opt.converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))


    accuracy = n_correct / float(max_iter * opt_batch_size)
    print('Rank: %d, Test loss: %f, accuray: %f' % (opt.rank, loss_avg.val(), accuracy))


def trainBatch(net, train_iter, criterion, optimizer, opt):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    loadData(opt.image, cpu_images)

    t, l = opt.converter.encode(cpu_texts)
    loadData(opt.text, t)
    loadData(opt.length, l)

    if opt.mlu:
        preds = net(opt.image.to(ct.mlu_device(), non_blocking=True))
    else:
        preds = net(opt.image)
    # if opt.rank != 0:
    #     preds = preds.to(0, non_blocking=True)  # warpctc_pytorch only correct on GPU 0?
    preds_size = Variable(torch.IntTensor([preds.size(1)] * batch_size))
    preds = preds.permute(1, 0, 2) # to use CTCloss format ?

    if opt.mlu:
        cost = torch.ops.torch_mlu.warp_ctc_loss(preds, opt.text.to(ct.mlu_device(), non_blocking=True), preds_size.to(ct.mlu_device(), non_blocking=True), opt.length.to(ct.mlu_device(), non_blocking=True), 0, 1, True, 0) / batch_size
    else:
        cost = criterion(preds, opt.text, preds_size, opt.length) / batch_size
    optimizer.zero_grad()

    if opt.mlu and opt.cnmix:
       with cnmix.scale_loss(cost, optimizer) as scaled_loss:
           scaled_loss.backward()
    else:
       cost.backward()

    optimizer.step()
    return cost

def train(net, train_loader, test_dataset, criterion, optimizer, loss_avg, time_avg, opt):
    print('Start train')
    epoch_size = len(train_loader)
    adaptive_cnt = int(os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT')) \
                   if (os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT') is not None) else 0
    if opt.iter > 0:
        epoch_size = opt.iter
    else:
        if opt.ddp:
            epoch_size = int(epoch_size / opt.ngpu)
    for epoch in range(opt.start_epoch + 1, opt.nepoch + 1):
        train_iter = iter(train_loader)
        i = 0
        batch_time_benchmark = []
        end = time.time()
        avg_loss = 0.0
        avg_time = 0.0
        metric_collector = MetricCollector(
                               enable_only_benchmark=True,
                               record_elapsed_time=True,
                               record_hardware_time=True if opt.mlu else False)
        metric_collector.place()
        while i < epoch_size:
            for p in net.parameters():
                p.requires_grad = True
            net.train()

            start_time = time.time()
            cost = trainBatch(net,train_iter, criterion, optimizer, opt)
            end_time = time.time() - start_time
            avg_loss += cost.item()
            metric_collector.record()
            metric_collector.place()
            avg_time += end_time
            loss_avg.add(cost)
            time_avg.add(end_time)

            i += 1

            if i % opt.displayInterval == 0:
                if opt.ddp:
                    print('Rank: %d [%d/%d][%d/%d] Loss: %f , Time: %f, throughout: %f' %
                        (opt.rank, epoch, opt.nepoch, i, epoch_size, loss_avg.val(), time_avg.val(), opt.batchSize/time_avg.val()))
                else:
                    print('[%d/%d][%d/%d] Loss: %f , Time: %f, throughout: %f' %
                        (epoch, opt.nepoch, i, epoch_size, loss_avg.val(), time_avg.val(), opt.batchSize/time_avg.val()))
                loss_avg.reset()
                time_avg.reset()
            # End 2 End time
            if i >= adaptive_cnt:
                batch_time_benchmark.append(time.time() - end)
            end = time.time()

        # insert and dump metrics
        metric_collector.insert_metrics(
                         net = "crnn",
                         batch_size = opt.batchSize,
                         precision = opt.opt_level if opt.cnmix else "fp32",
                         cards = opt.ngpu if opt.ddp else 1,
                         DPF_mode = "ddp" if opt.ddp else "single")
        if not opt.ddp or (opt.rank == 0):
            metric_collector.dump()

        if epoch % opt.saveInterval == 0:
            if opt.rank == 0:
                print('Rank0 saved model')
                if opt.cuda:
                    if not os.path.exists(opt.expr_dir):
                        os.makedirs(opt.expr_dir)
                    print("=> Save file to {}".format(opt.expr_dir))
                    if opt.ddp:
                        checkpoint = {"state_dict":net.module.state_dict(), "optimizer":optimizer.state_dict(),
                                      "epoch": epoch}
                    else:
                        checkpoint = {"state_dict":net.state_dict(), "optimizer":optimizer.state_dict(),
                                      "epoch": epoch}
                    torch.save(
                        checkpoint, '{0}/netCRNN_GPU_{1}.pth'.format(opt.expr_dir, epoch))
                    print("=> Model save finished")
                elif opt.mlu:
                    if not os.path.exists(opt.expr_dir):
                        os.makedirs(opt.expr_dir)
                    print("=> Save file to {}".format(opt.expr_dir))
                    if opt.ddp:
                        checkpoint = {"state_dict":net.module.state_dict(), "optimizer":optimizer.state_dict(),
                                      "epoch": epoch}
                    else:
                        checkpoint = {"state_dict":net.state_dict(), "optimizer":optimizer.state_dict(),
                                      "epoch": epoch}
                    if opt.cnmix:
                        checkpoint["cnmix"]=cnmix.state_dict()
                    torch.save(
                        checkpoint, '{0}/netCRNN_MLU_{1}.pth'.format(opt.expr_dir, epoch))
                    print("=> Model save finished")
                else:
                    torch.save(
                        net.state_dict(), '{0}/netCRNN_CPU_{1}.pth'.format(opt.expr_dir, epoch))
                    print("=> Model save finished")

    if opt.rank == 0:
        if opt.cuda:
            torch.save(
                net.state_dict(), '{0}/netCRNN_GPU_Final.pth'.format(opt.expr_dir))
        elif opt.mlu:
            torch.save(
                net.state_dict(), '{0}/netCRNN_MLU_Final.pth'.format(opt.expr_dir))
        else:
            torch.save(
                net.state_dict(), '{0}/netCRNN_CPU_Final.pth'.format(opt.expr_dir))


if __name__ == '__main__':
    main()

