# Resnet50v1_5 Networks for Image Classification in PyTorch

In this repository you will find implementations of resnet50 v1.5 models.

Detailed information can be found here:

## Model overview

The resnet50 v1.5 model is based on (https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets) repository.
This repository maintenance is performed on MLU devices for training and prediction.

#### Features

[MLU] performed on MLU devices for training and prediction.

[DDP] stands for DistributedDataParallel and is used for multi-MLU training.

## Setup
The following section lists the requirements in order to start training the Resnet50 v1.5 model.

### Requirements

* python >= 3.6 

* dllogger

## Quick Start Guide
1. Download and preprocess the dataset.

2. install dllogger.
pip --no-cache-dir --no-cache install git+https://github.com/NVIDIA/dllogger 

3. Start scrach training and inference.
train:     bash scratch_ngc_resnet50v1_5_4mlu.sh.
inference: python main.py $IMAGENET_TRAIN_DATASET --raport-file raport.json -j1 -p 100 --arch resnet50 -c fanin
           --workspace ${1:-./} -b 128 --epochs 91 --resume checkpoint.pth.tar --evaluate

3. Start training and inference.
bash cambricon/scratch_ngc_resnet50v1_5_4mlu.sh.

## Notice
In order to unify the calculation method of throughput(fps), changed the original logger
in resnet50v1_5/image_classification/training.py:

del(line 385-386):
    logger.log_metric("train.compute_ips", calc_ips(bs, it_time - data_time))
    logger.log_metric("train.total_ips", calc_ips(bs, it_time))
append:
    logger.log_metric("train.total_time", it_time)

del(line 390):
    logger.log_metric("train.total_ips", calc_ips(bs, it_time))
append:
    logger.log_metric("train.total_time", it_time)

