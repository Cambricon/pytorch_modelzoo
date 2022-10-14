# vision_classification(PyTorch)
## 模型概述
vision_classification系列网络是基于 [torchvision.models(v0.7.0)](https://github.com/pytorch/vision/tree/v0.7.0/torchvision/models)的寒武纪实现版本，torchvision是pytorch用来处理图像，构建计算机视觉模型的算法库，其中torchvision.models有包含常用的算法模型结构，且含预训练模型，例如ResNet50、VGG16、AlexNet等；

## 支持情况
### 训练模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50  | PyTorch  | MLU370-X8  | AMP/FP32  | Yes  | Not Tested
ResNet18  | PyTorch  | MLU370-X8  | FP32  | Yes  | Not Tested
VGG16  | PyTorch  | MLU370-X8  | AMP/FP32  | Yes  | Not Tested
MobileNetv2  | PyTorch  | MLU370-X8  | AMP/FP32  | Yes  | Not Tested
AlexNet  | PyTorch  | MLU370-X8  | FP32  | Yes  | Not Tested
GoogLeNet  | PyTorch  | MLU370-X8  | AMP/FP32  | Yes  | Not Tested
ResNet101  | PyTorch  | MLU370-X8  | FP32  | Yes  | Not Tested
VGG19  | PyTorch  | MLU370-X8  | FP32  | Yes  | Not Tested
VGG16_bn  | PyTorch  | MLU370-X8  | FP32  | Yes  | Not Tested
ShuffleNet_v2_x0_5  | PyTorch  | MLU370-X8  | FP32  | Yes  | Not Tested
ShuffleNet_v2_x1_0  | PyTorch  | MLU370-X8  | FP32  | Yes  | Not Tested

### 推理模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision   | Supported Infer Mode | 
----- | ----- | ----- | ----- | ----- |
ResNet50  | PyTorch  | MLU370-S4/X4  | FP16/FP32  | torch2mm/cnnl | 
ResNet18  | PyTorch  | MLU370-S4/X4  | FP16/FP32  | torch2mm/cnnl | 
VGG16  | PyTorch  | MLU370-S4/X4  | FP16/FP32  | torch2mm/cnnl | 
MobileNetv2  | PyTorch  | MLU370-S4/X4  | FP16/FP32  | torch2mm/cnnl |
AlexNet  | PyTorch  | MLU370-S4/X4  | FP16/FP32  | torch2mm/cnnl |
GoogLeNet  | PyTorch  | MLU370-S4/X4  | FP16/FP32  | torch2mm/cnnl |
ResNet101  | PyTorch  | MLU370-S4/X4  | FP16/FP32  | torch2mm/cnnl |
VGG19  | PyTorch  | MLU370-S4/X4  | FP16/FP32  | torch2mm/cnnl |
VGG16_bn  | PyTorch  | MLU370-S4/X4  | FP16/FP32  | torch2mm/cnnl |
ShuffleNet_v2_x0_5  | PyTorch  | MLU370-S4/X4  | FP16/FP32  | torch2mm/cnnl |
ShuffleNet_v2_x1_0  | PyTorch  | MLU370-S4/X4  | FP16/FP32  | torch2mm/cnnl |


## 默认参数配置
### 模型训练默认参数配置
以下为vision_classification模型的默认参数配置：

### Optimizer
Models  | 优化器  | Learning Rate   | Learning rate schedule |  Weight decay | Label Smoothing | Epoch
---- | ----- | ----- | ----- | ----- | ----- |---- |
ResNet50  | SGD  | 0.1  | Linear schedule  | 1e-4 | None | 100
ResNet18  | SGD  | 0.2  | Linear schedule  | 1e-4 | None | 150
VGG16  | SGD  | 0.02  | Linear schedule  | 1e-4 | None | 100
MobileNetv2  | SGD  | 0.05  | Linear schedule  | 0.00004 | None | 100
AlexNet  | SGD  | 0.04  | Linear schedule  | 1e-4 | None | 150
GoogLeNet  | SGD  | 0.1  | Linear schedule  | 4e-5 | None | 150
ResNet101  | SGD  | 0.1  | Linear schedule  | 4e-5 | None | 100
VGG19  | SGD  | 0.01  | Linear schedule  | 1e-4 | None | 100
VGG16_bn  | SGD  | 0.01  | Linear schedule  | 1e-4 | None | 150
ShuffleNet_v2_x0_5  | SGD  | 0.5  | Linear schedule  | 4e-5 | None | 300
ShuffleNet_v2_x1_0  | SGD  | 0.5  | Linear schedule  | 4e-5 | None | 300

### Data Augmentation
模型使用了以下数据增强方法：
* 训练
    * Normolization
    * Crop image to 224*224
    * RandomHorizontalFlip
* 验证
    * Normolization
    * Crop image to 256*256
    * Center crop to 224*224

### 模型推理默认参数配置
* modeldir: 没有指定情况下，默认使用 torchvision 的预训练模型，可选指定训练完成的权重(eg. --modeldir /model/xxxx.pt)
* batch_size: 64
* input_data_type：默认使用 float32
* jit&jit_fuse：默认开启
* qint：默认不开启量化


## 依赖项检查
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算版本MLU370-X8;
* Cambricon Driver >=v4.20.6；
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## Quick Start Guide

### 数据集准备
该vision_classification系列模型基于ImageNet1K训练，数据集下载：<https://www.image-net.org/>。数据集请放在` $IMAGENET_TRAIN_DATASET`目录下。目录结构为：
```
├── train
│   ├── n01440764
│   ├── n01443537
│   ├── ...
├── train.txt
├── val
│   ├── n01440764
│   ├── n01443537
│   ├── ...
└── val.txt
```


### 环境准备
#### 基于base docker image安装
##### 1、导入镜像
```
##下载Cambricon PyTorch docker镜像
docker load -i xxx.tar.gz
```

##### 2、启动测试容器
```
bash run_docker.sh
```

##### 3、启动虚拟环境，安装依赖，并设置环境变量

```
source /torch/venv3/bin/activate
pip install -r requirement.txt
source env.sh
```

#### 使用Dockerfile准备环境
#### 1、生成vision_classification的Docker镜像：

```
export IMAGE_NAME=test_vision_classification
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../
```

####  2、创建容器

```
docker run -it --ipc=host -v /data:/data --device /dev/cambricon_ctl --privileged --name test_classify --network=host $IMAGE_NAME
```

##### 3、启动虚拟环境，设置环境变量

```
source /torch/venv3/bin/activate
source env.sh
```

### Run 脚本执行
```
bash run_scripts/ResNet50/MLU370_ResNet50_AMP_100E_4MLUs_Train.sh
```

#### 一键执行训练脚本
Models  | Framework  | MLU   | MODE  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50  | PyTorch  | MLU370-X8  |  AMP(from scratch)  | 4  | bash run_scripts/ResNet50/MLU370_ResNet50_AMP_100E_4MLUs_Train.sh
ResNet50  | PyTorch  | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/ResNet50/MLU370_ResNet50_FP32_100E_4MLUs_Train.sh
ResNet18  | PyTorch  | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/ResNet18/MLU370_ResNet18_FP32_100E_4MLUs_Train.sh
VGG16  | PyTorch  | MLU370-X8  |  AMP(from scratch)  | 4  | bash run_scripts/VGG16/MLU370_VGG16_AMP_100E_4MLUs_Train.sh
VGG16  | PyTorch  | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/VGG16/MLU370_VGG16_FP32_100E_4MLUs_Train.sh
MobileNet_v2  | PyTorch  | MLU370-X8  |  AMP(from scratch)  | 4  | bash run_scripts/MobileNet_v2/MLU370_MobileNetv2_AMP_150E_4MLUs_Train.sh
MobileNet_v2  | PyTorch  | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/MobileNet_v2/MLU370_MobileNetv2_FP32_150E_4MLUs_Train.sh
AlexNet  | PyTorch  | MLU370-X8  | FP32(from scratch)  | 8  | bash run_scripts/AlexNet/MLU370_AlexNet_FP32_100E_4MLUs_Train.sh
GoogleNet  | PyTorch  | MLU370-X8  |  AMP(from scratch)  | 4  | bash run_scripts/GoogleNet/MLU370_GoogleNet_AMP_150E_4MLUs_Train.sh
GoogleNet  | PyTorch  | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/GoogleNet/MLU370_GoogleNet_FP32_150E_4MLUs_Train.sh
ResNet101  | PyTorch  | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/ResNet101/MLU370_ResNet101_FP32_100E_4MLUs_Train.sh
VGG19  | PyTorch  | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/VGG19/MLU370_VGG19_FP32_100E_4MLUs_Train.sh
VGG16_bn  | PyTorch  | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/VGG16_bn/MLU370_VGG16_bn_FP32_100E_4MLUs_Train.sh
ShuffleNet_v2_x0_5  | PyTorch  | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/ShuffleNet_v2_x0_5/MLU370_ShuffleNetv2x05_FP32_300E_4MLUs_Train.sh
ShuffleNet_v2_x1_0  | PyTorch  | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/ShuffleNet_v2_x1_0/MLU370_ShuffleNetv2x10_FP32_300E_4MLUs_Train.sh








#### 一键执行推理脚本
Models  | Framework  | MLU   |Run
----- | ----- | ----- | ----- | 
ResNet50  | PyTorch  | MLU370-S4/MLU370-X4 | bash run_scripts/ResNet50/MLU370_ResNet50_Infer.sh
ResNet18  | PyTorch  | MLU370-S4/MLU370-X4 | bash run_scripts/ResNet18/MLU370_ResNet18_Infer.sh
VGG16  | PyTorch  | MLU370-S4/MLU370-X4 | bash run_scripts/VGG16/MLU370_VGG16_Infer.sh
MobileNetv2  | PyTorch  | MLU370-S4/MLU370-X4 | bash run_scripts/MobileNet_v2/MLU370_MobileNetv2_Infer.sh
AlexNet  | PyTorch  | MLU370-S4/MLU370-X4 | bash run_scripts/AlexNet/MLU370_AlexNet_Infer.sh
GoogleNet  | PyTorch  | MLU370-S4/MLU370-X4 | bash run_scripts/GoogleNet/MLU370_GoogleNet_Infer.sh
ResNet101  | PyTorch  | MLU370-S4/MLU370-X4 | bash run_scripts/ResNet101/MLU370_ResNet101_Infer.sh
VGG19  | PyTorch  | MLU370-S4/MLU370-X4 | bash run_scripts/VGG19/MLU370_VGG19_Infer.sh
VGG16_bn  | PyTorch  | MLU370-S4/MLU370-X4 | bash run_scripts/VGG16_bn/MLU370_VGG16_bn_Infer.sh
ShuffleNet_v2_x0_5  | PyTorch  | MLU370-S4/MLU370-X4 | bash run_scripts/ShuffleNet_v2_x0_5/MLU370_ShuffleNetv2x05_Infer.sh
ShuffleNet_v2_x1_0  | PyTorch  | MLU370-S4/MLU370-X4 | bash run_scripts/ShuffleNet_v2_x1_0/MLU370_ShuffleNetv2x10_Infer.sh


### 命令行选项运行

#### 训练 classify_train.py 所有可选参数如下：

`python classify_train.py -h`

```
usage: classify_train.py [-h] [-p N] [-m DIR] [--data DIR] [-j N] [--epochs N]
                         [--start-epoch N] [-b N] [--lr LR] [--momentum M]
                         [--wd W] [--resume_multi_device] [--resume PATH] [-e]
                         [--world-size WORLD_SIZE] [--rank RANK]
                         [--dist-url DIST_URL] [--dist-backend DIST_BACKEND]
                         [--seed SEED] [--save_ckp] [--iters N]
                         [--device DEVICE] [--device_id DEVICE_ID]
                         [--pretrained] [--multiprocessing-distributed]
                         [--ckpdir DIR] [--logdir DIR] [--hvd HVD] [--cnmix]
                         [--opt_level OPT_LEVEL] [--dummy_test] [--pyamp]
                         [--start_eval_at START_EVAL_AT]
                         [--evaluate_every EVALUATE_EVERY]
                         [--quality_threshold QUALITY_THRESHOLD]

PyTorch ImageNet Training

optional arguments:
  -h, --help            show this help message and exit
  -p N, --print-freq N  print frequency (default: 1)
  -m DIR, --modeldir DIR
                        path to dir of models and mlu operators, default is ./
                        and from torchvision
  --data DIR            path to dataset
  -j N, --workers N     number of data loading works (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  --resume_multi_device
                        Only when model is saved by gpu distributed, enable
                        this to load model with submodule
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --save_ckp            Enable save checkpoint
  --iters N             iters per epoch
  --device DEVICE       Use cpu gpu or mlu device
  --device_id DEVICE_ID
                        Use specified device for training, useless in
                        multiprocessing distributed training
  --pretrained          Use a pretrained model
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
  --ckpdir DIR          Where to save ckps
  --logdir DIR          Where to save logs
  --hvd HVD             how manys cards if using horovod
  --cnmix               use cnmix for mixed precision training
  --opt_level OPT_LEVEL
                        choose level of mixing precision
  --dummy_test          use fake data to traing
  --pyamp               use pytorch amp for mixed precision training
  --start_eval_at START_EVAL_AT
                        start evaluation at specified epoch
  --evaluate_every EVALUATE_EVERY, --eval_every EVALUATE_EVERY
                        evaluate at every epochs
  --quality_threshold QUALITY_THRESHOLD
                        target accuracy
```

#### 推理 classif_infer.py 所有可选参数如下：

`python classify_infer.py -h`

```
usage: classify_infer.py [-h] [--batch_size BATCH_SIZE]
                         [--input_data_type {float32,float16}]
                         [--network NETWORK] [-j N] [--iters ITERS]
                         [--warmup_iters WARMUP_ITERS]
                         [--device {cpu,mlu,gpu}] [--data DATA] [-p N]
                         [--seed SEED] [--qint {int8,int16,no_quant}]
                         [--quant_batch_num QUANT_BATCH_NUM]
                         [--fusion_backend {no,torch2trt,torch2mm}]
                         [--ckpt CKPT] [--only_genoff ONLY_GENOFF]
                         [--offline_model_path OFFLINE_MODEL_PATH]
                         [--do_benchmark DO_BENCHMARK]

Pre-checkin and Daily test script.

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size for one inference.
  --input_data_type {float32,float16}
  --network NETWORK     the network that will be running.
  -j N, --workers N     number of data loading works (default: 4)
  --iters ITERS
  --warmup_iters WARMUP_ITERS
  --device {cpu,mlu,gpu}
  --data DATA           imagenet validation dir
  -p N, --print-freq N  print frequency (default: 1)
  --seed SEED           seed for initializing training.
  --qint {int8,int16,no_quant}
  --quant_batch_num QUANT_BATCH_NUM
                        Set image numbers to evaluate quantized params,
                        default is 5.
  --fusion_backend {no,torch2trt,torch2mm}
  --ckpt CKPT           model checkpoint file
  --only_genoff ONLY_GENOFF
                        only generate torchscript model
  --offline_model_path OFFLINE_MODEL_PATH
                        torchscript offline model path
  --do_benchmark DO_BENCHMARK
                        do benchmark test
```

## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

ImageNet1K 数据集下载链接：https://www.image-net.org/       \
torchvision.models 模型代码链接：https://github.com/pytorch/vision/tree/v0.4.1/torchvision/models


## Release_Notes
@TODO
