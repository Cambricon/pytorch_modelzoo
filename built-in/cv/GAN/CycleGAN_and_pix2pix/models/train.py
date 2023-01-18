"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

import torch
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../../tools/utils/")
from metric import MetricCollector

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

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    if opt.seed >= 0:  # set random seed if seed > 0
        torch.manual_seed(opt.seed)
        import numpy as np
        np.random.seed(opt.seed)
    if opt.device == "gpu" and torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        if opt.seed >= 0:
            cudnn.deterministic = True  # cuDNN to only use deterministic convolution algorithms.
            cudnn.benchmark = False  # cuDNN deterministically select an convolution algorithm.
        try:
            if torch.backends.cuda.matmul.allow_tf32 == True or cudnn.allow_tf32 == True:
                torch.backends.cuda.matmul.allow_tf32 = False  # set FP32 mode
                cudnn.allow_tf32 = False  # set FP32 mode
                print("TF32 is modified False, start FP32 Running")
        except Exception as e:
            print(e)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    # for internal benchmark test
    enable_only_benchmark = True if "BENCHMARK_LOG" in os.environ else False
    enable_only_avglog = True if "AVG_LOG" in os.environ else False
    metric_collector = MetricCollector(
            enable_only_benchmark=enable_only_benchmark,
            enable_only_avglog=enable_only_avglog,
            record_elapsed_time=True,
            record_hardware_time=True if opt.device == "mlu" else False)
    metric_collector.place()
    iters = 0
    loss_G_GAN_meter = AverageMeter()
    loss_G_L1_meter = AverageMeter()
    loss_D_real_meter = AverageMeter()
    loss_D_fake_meter = AverageMeter()
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        if iters == opt.iters:
            break
        for i, data in enumerate(dataset):  # inner loop within one epoch
            if iters == opt.iters:
                break
            iters += 1
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            losses = model.get_current_losses()
            loss_G_GAN = losses["G_GAN"]
            loss_G_L1 = losses["G_L1"]
            loss_D_real = losses["D_real"]
            loss_D_fake = losses["D_fake"]
            loss_G_GAN_meter.update(loss_G_GAN)
            loss_G_L1_meter.update(loss_G_L1)
            loss_D_real_meter.update(loss_D_real)
            loss_D_fake_meter.update(loss_D_fake)
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
            # MetricCollector record
            metric_collector.record()
            metric_collector.place()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    precision = "fp32"
    # insert metrics and dump metrics
    metric_collector.insert_metrics(
        net = "pix2pix",
        batch_size = opt.batch_size,
        precision = precision,
        cards = 1,
        DPF_mode = "single",
        accuracy = [["G_GAN:", round(loss_G_GAN_meter.avg,3)],
                    ["G_L1:", round(loss_G_L1_meter.avg,3)],
                    ["D_real:", round(loss_D_real_meter.avg,3)],
                    ["D_fake:", round(loss_D_fake_meter.avg,3)]]
        )
    metric_collector.dump()