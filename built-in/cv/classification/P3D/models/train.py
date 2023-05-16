from __future__ import print_function
import copy
import os
import sys
import os.path
import re
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
import torch.optim as optim
import torch.utils.data as data
from meter import AverageMeter
from logger import Logger
from transforms import *
from Dataset import MyDataset
from models.p3d_model import P3D199, get_optim_policies
from models.C3D import C3D
from models.i3dpt import I3D
from utils import check_gpu, transfer_model, accuracy, get_learning_rate
from lr_scheduler import CyclicLR
from torch.cuda.amp import autocast, GradScaler

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../tools/utils/")
from metric import MetricCollector

try:
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
except ImportError:
    print("without torch_mlu")

try:
   import cnmix
except ImportError:
   print("train without cnmix!")

def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt
class dummy_data_loader():
    def __init__(self, len = 0, images_size = (3, 224, 224), batch_size = 1, num_classes = 1000):
        self.len = len
        images = torch.normal(mean = 0.485 , std = 0.229, size = (batch_size,)+images_size)
        target = torch.randint(low = 0, high = num_classes, size = (batch_size,))
        self.images = images.to(ct.mlu_device(), non_blocking=True)
        self.target = target.to(ct.mlu_device(), non_blocking=True)
        self.data = 0
    def __iter__(self):
        return self
    def __len__(self):
        return self.len
    def __next__(self):
        if self.data > self.len:
            raise StopIteration
        else:
            self.data += 1
            return self.images, self.target

class Training(object):
    def __init__(self, name_list, num_classes=400, modality='RGB', **kwargs):
        self.__dict__.update(kwargs)
        self.num_classes = num_classes
        self.modality = modality
        self.name_list = name_list
        # set accuracy avg = 0
        self.count_early_stop = 0
        # Set best precision = 0
        self.best_prec1 = 0
        # init start epoch = 0
        self.start_epoch = 0
        if self.local_rank == 0:
            print("parameters: ", self.__dict__)
        self.checkDataFolder()

        self.train_loader, self.val_loader = self.loading_data()

        self.loading_model()

        # run
        self.processing()
        if self.random:
            print('random pick images')

    def check_early_stop(self, accuracy, logger, start_time):
        if self.best_prec1 <= accuracy:
            self.count_early_stop = 0
        else:
            self.count_early_stop += 1

        if self.count_early_stop > self.early_stop:
            print('Early stop')
            end_time = time.time()
            print("--- Total training time %s seconds ---" %
                  (end_time - start_time))
            logger.info("--- Total training time %s seconds ---" %
                        (end_time - start_time))
            exit()

    def checkDataFolder(self):
        dir = self.logdir + self.model_type + '_' + self.data_set
        if not os.path.exists(dir):
            try:
                os.makedirs(dir)
            except:
                print("INFO: Multiprocesses make dir: ", dir)
        self.data_folder = dir

    # Loading P3D model
    def loading_model(self):

        print('Loading %s model' % (self.model_type))

        if self.model_type == 'C3D':
            self.model = C3D()
            if self.pretrained:
                self.model.load_state_dict(torch.load('c3d.pickle'))
        elif self.model_type == 'I3D':
            if self.pretrained:
                self.model = I3D(num_classes=400, modality='rgb')
                self.model.load_state_dict(
                    torch.load('kinetics_i3d_model_rgb.pth'))
            else:
                self.model = I3D(num_classes=self.num_classes, modality='rgb')
        else:
            if self.pretrained:
                print("=> using pre-trained model")
                self.model = P3D199(
                    pretrained=True, num_classes=400, dropout=self.dropout)

            else:
                print("=> creating model P3D")
                self.model = P3D199(
                    pretrained=False, num_classes=400, dropout=self.dropout)
        # Transfer classes
        self.model = transfer_model(model=self.model, model_type=self.model_type, num_classes=self.num_classes)

        # optionally resume from a checkpoint
        if self.resume:
            if os.path.isfile(self.resume):
                print("=> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume, map_location='cpu')
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'], strict=True if self.device_param=='gpu' else False)
                resume_optimizer = checkpoint['optimizer']
                if self.pyamp and isinstance(checkpoint, dict) and 'amp' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['amp'])

                print("=> loaded checkpoint '{}' (epoch {})".format(
                    self.resume, checkpoint['epoch']))
            else:
                print("=> resume no checkpoint found at '{}'".format(self.resume))
                exit()

        if self.evaluate:
            if os.path.isfile(self.evaluate):
                print("=> loading checkpoint '{}'".format(self.evaluate))
                checkpoint = torch.load(self.evaluate, map_location='cpu')
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                print("best prec1: ", self.best_prec1)
                self.model.load_state_dict(checkpoint['state_dict'])
                eval_optimizer = checkpoint['optimizer']
                print("=> loaded checkpoint '{}' (epoch {})".format(
                    self.evaluate, checkpoint['epoch']))
            else:
                print("=> eval no checkpoint found at '{}'".format(self.evaluate))
                exit()
        if self.device_param == 'mlu':
            self.model.to(ct.mlu_device())
        elif self.device_param == 'gpu':
            self.model.to(torch.device("cuda"))
        # define loss function (criterion) and optimizer
        params = self.model.parameters()
        if self.model_type == 'P3D':
            params = get_optim_policies(model=self.model, modality=self.modality, enable_pbn=True)

        self.optimizer = optim.SGD(params=params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        if self.resume: # Resume optimizer
            self.optimizer.load_state_dict(resume_optimizer)
        if self.evaluate: # Resume optimizer
            self.optimizer.load_state_dict(eval_optimizer)

        if self.device_param == 'mlu' and self.cnmix:
            self.model, self.optimizer = cnmix.initialize(self.model, self.optimizer, opt_level=self.opt_level)
            if self.resume and os.path.isfile(self.resume):
                checkpoint = torch.load(self.resume, map_location='cpu')
                if isinstance(checkpoint, dict) and 'cnmix' in checkpoint:
                    cnmix.load_state_dict(checkpoint['cnmix'])

        # Check gpu and run parallel
        if self.device_param == "gpu":
            if self.distributed:
                self.model = DDP(self.model, device_ids=[self.local_rank])
            else:
                self.model = torch.nn.DataParallel(self.model).cuda()

        else:
            if self.distributed:
                self.model = DDP(self.model, device_ids=[self.local_rank])
            print("using MLU")

        if self.device_param == "gpu":
            self.criterion = nn.CrossEntropyLoss().cuda()
        else:
            self.criterion = nn.CrossEntropyLoss().to("mlu")
            ct.to(self.optimizer, torch.device('mlu'))

        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='min', patience=5, verbose=True)

        if self.device_param == 'mlu' and self.cnmix:
            cnmix.cnmix_set_amp_quantify_params('all', {'batch_size': self.batch_size,
                                                        'data_num': self.batch_size * len(self.train_loader)})
        cudnn.benchmark = False
        cudnn.deterministic = True

    # Loading data
    def loading_data(self):
        random = True if self.random else False
        size = 160
        if self.model_type == 'C3D':
            size = 112
        if self.model_type == 'I3D':
            size = 224

        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
        train_transformations = Compose([
            RandomSizedCrop(size),
            RandomHorizontalFlip(),
            # Resize((size, size)),
            # Resize(256),
            # ColorJitter(
            #     brightness=0.4,
            #     contrast=0.4,
            #     saturation=0.4,
            # ),
            ToTensor(),
            normalize])

        val_transformations = Compose([
            # Resize((182, 242)),
            # Resize(256),
            CenterCrop(size),
            ToTensor(),
            normalize
        ])

        train_dataset = MyDataset(
            self.data,
            data_folder="train",
            name_list=self.name_list,
            version="1",
            transform=train_transformations,
            num_frames=self.num_frames,
            random=random
        )
        print("train size: ", len(train_dataset))

        val_dataset = MyDataset(
            self.data,
            data_folder="validation",
            name_list=self.name_list,
            version="1",
            transform=val_transformations,
            num_frames=self.num_frames,
            random=random
        )

        sampler = None
        if self.distributed:
            ndevs_per_node=int(os.environ['WORLD_SIZE'])
            if self.device_param == "mlu":
                sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas = ndevs_per_node, rank = self.local_rank)
            else:
                print("distributed sampler")
                sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=sampler is None, #True for none distributed,
            num_workers=self.workers,
            pin_memory=True,
            sampler=sampler,)

        val_loader = data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=False)
        if self.dummy_test:
            train_loader = dummy_data_loader( len(train_loader),
                                              images_size = (3, self.num_frames, size, size),
                                              batch_size = self.batch_size,
                                              num_classes = 100)
        return (train_loader, val_loader)

    def processing(self):
        log_file = os.path.join(self.data_folder, 'train_p3d.log')

        logger = Logger('train', log_file)

        iters = len(self.train_loader)

        step_size = iters * 2
        self.scheduler = CyclicLR(optimizer=self.optimizer, step_size=step_size, base_lr=self.lr, max_lr=6 * self.lr)

        if self.evaluate:
            self.validate(logger)
            return

        iter_per_epoch = len(self.train_loader)
        logger.info('Iterations per epoch: {0}'.format(iter_per_epoch))
        print('Iterations per epoch: {0}'.format(iter_per_epoch))
        print('Iterations per epoch validate: {0}'.format(len(self.val_loader)))

        start_time = time.time()
        if self.train_steps != -1:
            run_eps = self.epochs - self.start_epoch
            self.epochs = self.start_epoch + min (run_eps, (self.train_steps + iter_per_epoch -1)//iter_per_epoch)
        scaler = None
        if self.pyamp:
            self.scaler = GradScaler()

        for epoch in range(self.start_epoch, self.epochs):
            if self.distributed and not self.dummy_test:
                self.train_loader.sampler.set_epoch(epoch)
            # self.adjust_learning_rate(epoch)

            # train for one epoch
            train_losses, train_acc = self.train(logger, epoch)

            # evaluate on validation set
            with torch.no_grad():
                val_losses, val_acc = self.validate(logger)

            # self.scheduler.step(val_losses.avg)

            # remember best Accuracy and save checkpoint
            is_best = val_acc.avg > self.best_prec1
            self.best_prec1 = max(val_acc.avg, self.best_prec1)

            if self.local_rank == 0:
                save_params={
                   'epoch': epoch + 1,
                   'state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
                   'best_prec1': self.best_prec1,
                   'optimizer' : self.optimizer.state_dict(),
               }
                if self.device_param == "mlu" and self.cnmix:
                   save_params['cnmix'] = cnmix.state_dict()
                if self.pyamp:
                    save_params['amp'] = self.scaler.state_dict()
                self.save_checkpoint(save_params, is_best, epoch + 1)


            self.check_early_stop(val_acc.avg, logger, start_time)
        end_time = time.time()
        print("--- Total training time %s seconds ---" %
              (end_time - start_time))
        logger.info("--- Total training time %s seconds ---" %
                    (end_time - start_time))

    # Training
    def train(self, logger, epoch):
        adaptive_cnt = int(os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT')) if (os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT') is not None) else 0
        batch_time_benchmark = []
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        rate = get_learning_rate(self.optimizer)[0]
        # switch to train mode
        self.model.train()
        end = time.time()

        # for internal benchmark test
        metric_collector = MetricCollector(
            enable_only_benchmark=True,
            record_elapsed_time=True,
            record_hardware_time=True if self.device_param == 'mlu' else False)
        metric_collector.place()

        for i, (images, target) in enumerate(self.train_loader):
            if i + (epoch - self.start_epoch) * len(self.train_loader) == self.train_steps:
                break
            last_batch = i == len(self.train_loader)-1
            # adjust learning rate scheduler step
            self.scheduler.batch_step()
            # measure data loading time
            data_time.update(time.time() - end)
            if not self.dummy_test:
                if self.device_param == "gpu":
                    images = images.cuda()
                    target = target.cuda()
                else:
                    images = images.to('mlu', non_blocking=True)
                    target = target.to('mlu', non_blocking=True)
            image_var = torch.autograd.Variable(images)
            label_var = torch.autograd.Variable(target)

            self.optimizer.zero_grad()

            with autocast(enabled=self.pyamp):
                # compute y_pred
                y_pred = self.model(image_var)
                if self.model_type == 'I3D':
                    y_pred = y_pred[0]

                loss = self.criterion(y_pred, label_var)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(y_pred.data, target, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            acc.update(prec1.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))
            # compute gradient and do SGD step

            if self.device_param == "mlu" and self.cnmix:
                with cnmix.scale_loss(loss,self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif self.pyamp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if self.pyamp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.device_param == "gpu":
                torch.cuda.synchronize()
            elif self.device_param == "mlu":
                torch.mlu.synchronize()

            # MetricCollector record
            metric_collector.record()
            metric_collector.place()

            # measure elapsed time
            if i + (epoch - self.start_epoch) * len(self.train_loader) >= adaptive_cnt:
                batch_time_benchmark.append(time.time() - end)
            batch_time.update(time.time() - end)
            end = time.time()
            if last_batch or i % self.print_freq == 0:
                print('Process: {0}\t'
                      'Epoch: [{1}/{2}][{3}/{4}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Lr {rate:.5f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(self.local_rank,
                                                                      epoch, self.epochs, i, len(self.train_loader),
                                                                      batch_time=batch_time, data_time=data_time,
                                                                      rate=rate,
                                                                      loss=losses, top1=top1, top5=top5))

        # insert metrics and dump metrics
        if self.pyamp:
            precision = "amp"
        elif self.cnmix:
            precision = self.opt_level
        else:
            precision = "fp32"
        metric_collector.insert_metrics(
            net = "P3D",
            batch_size = self.batch_size,
            precision = precision,
            cards = self.world_size if self.distributed else 1,
            DPF_mode = "ddp " if self.distributed else "single")

        if ((self.distributed == False) or (self.rank == 0)):
            metric_collector.dump()

        logger.info('Process: {0}\t'
                    'Epoch: [{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Lr {rate:.5f}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(self.local_rank,
                                                                    epoch, self.epochs,
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, rate=rate, loss=losses,
                                                                    top1=top1,
                                                                    top5=top5))
        return losses, acc

    # Validation
    def validate(self, logger):
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        for i, (images, labels) in enumerate(self.val_loader):
            if i == self.eval_steps:
                break
            if self.device_param == "gpu":
                images = images.cuda()
                labels = labels.cuda()
            else:
                images = images.to('mlu', non_blocking=True)
                labels = labels.to('mlu', non_blocking=True)

            image_var = torch.autograd.Variable(images)
            label_var = torch.autograd.Variable(labels)

            # compute y_pred
            y_pred = self.model(image_var)
            if self.model_type == 'I3D':
                y_pred = y_pred[0]

            loss = self.criterion(y_pred, label_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(y_pred.data, labels, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            acc.update(prec1.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        metric_collector = MetricCollector(enable_only_avglog=True)
        metric_collector.insert_metrics(net = "P3D",
                                        accuracy = [top1.avg, top5.avg])
        if ((self.distributed == False) or (self.rank == 0)):
            metric_collector.dump()
            print(
                '* validate Process: {local_rank} Accuracy {acc.avg:.3f}  Loss {loss.avg:.3f}'.format(
                    local_rank=self.local_rank, acc=acc, loss=losses))
            logger.info(
                '* validate Process: {local_rank} Accuracy {acc.avg:.3f}  Loss {loss.avg:.3f}'.format(
                    local_rank=self.local_rank, acc=acc, loss=losses))

        return losses, acc

    # save checkpoint to file
    def save_checkpoint(self, state, is_best, id):
        checkpoint = os.path.join(self.data_folder, str(id)+'ckp_p3d.pth.tar')
        torch.save(state, checkpoint)
        model_best = os.path.join(self.data_folder, 'model_best_p3d.pth.tar')
        if is_best:
            shutil.copyfile(checkpoint, model_best)

    # adjust learning rate for each epoch
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 3K iterations"""
        iters = len(self.train_loader)
        num_epochs = 3000 // iters
        decay = 0.1 ** (epoch // num_epochs)
        lr = self.lr * decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = decay * param_group['decay_mult']
