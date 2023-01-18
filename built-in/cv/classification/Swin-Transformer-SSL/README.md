# Swin-Transformer-SSL(Pytorch)
---
## 模型概述
  Swin-Transformer-SSL网络源于论文[Self-Supervised Learning with Swin Transformers](https://arxiv.org/pdf/2105.04553)。Swin-Transformer-SSL网络的完整训练需要两阶段训练，第一阶段是自监督预训练(Self-Supervised Pre-training)，第二阶段是Linear Evaluation训练，第一阶段训练得到的权重作为第二阶段的初始权重。

  本仓库为Swin-Transformer-SSL的MLU实现，具体网络结构为 MoBY Swin-T，GPU实现可参考仓库: [Transformer-SSL](https://github.com/SwinTransformer/Transformer-SSL)

## 支持情况
---
### 训练模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-MLUs |
----- | ----- | ----- | ----- | ----- |
Swin-Transformer-SSL  | PyTorch1.9  | MLU370-X8  | AMP/FP32  | Yes  |

### 推理模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision   | Supported Infer Mode | 
----- | ----- | ----- | ----- | ----- |
Swin-Transformer-SSL  | PyTorch1.9  | MLU370-X8  | FP32  | CNNL |

## 默认参数配置
---
### Optimizer
Models  | 优化器  | Learning Rate   | Learning rate schedule |  Weight decay | Epoch
---- | ----- | ----- | ----- | ----- | ----- |
Swin-Transformer-SSL <br> (Self-Supervised Pre-training)  | AdamW  | 5e-4  | ConsineLR schedule  | 0.05 | 300
Swin-Transformer-SSL <br> (Linear Evaluation)  | SGD  | 0.5  | ConsineLR schedule  | 0 | 100

### Data Augmentation
模型使用了以下数据增强方法：

第一阶段训练(Self-Supervised Pre-training)
* RandomResizedCrop
* RandomHorizontalFlip
* ColorJitter
* GaussianBlur
* Normolization

第二阶段训练(Linear Evaluation)
* RandomResizedCrop
* RandomHorizontalFlip
* Normolization

推理
* Resize
* CenterCrop
* Normolization

## 环境依赖
---
- Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
- 服务器装配好寒武纪计算板卡MLU370-X8;
- Cambricon Driver >=v4.20.6；
- CNToolKit >=2.8.3;
- CNNL >=1.10.2;
- CNCL >=1.1.1;
- CNLight >=0.12.0;
- CNPyTorch >= 1.3.0;
- 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 快速入门指南
---
### 文件说明
- run_scripts/ 包含一键训练和推理的shell脚本文件
- models/ 包含原始模型仓库文件
- `moby_main.py` 模型第一阶段训练入口，更多信息使用`python moby_main.py -h`查看
- `moby_linear.py` 模型第二阶段训练&推理入口，更多信息使用`python moby_linear.py -h`查看

### 准备数据集
#### 方式一(推荐)：
下载数据集[ImageNet-1k](http://image-net.org/)，并解压。解压后的数据集请放在` $IMAGENET_TRAIN_DATASET`目录下(`IMAGENET_TRAIN_DATASET`是需要设置的环境变量)，目录结构为：
  ```bash 
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
指定数据集路径：
```bash
export IMAGENET_TRAIN_DATASET=/path/to/dataset
```

#### 方式二：
为了提高从大量小文件中读取图像的速度，还支持压缩的 ImageNet。
下载数据集[ImageNet-1k](http://image-net.org/)，无需解压。数据集请放在`$IMAGENET_TRAIN_DATASET`目录下，目录结构为：
```bash
├── train_map.txt
├── train.zip
├── val_map.txt
└── val.zip
```

val_map.txt 和 train_map.txt中的部分内容如下：
```
$ head -n 5 data/ImageNet-Zip/val_map.txt
ILSVRC2012_val_00000001.JPEG	65
ILSVRC2012_val_00000002.JPEG	970
ILSVRC2012_val_00000003.JPEG	230
ILSVRC2012_val_00000004.JPEG	809
ILSVRC2012_val_00000005.JPEG	516

$ head -n 5 data/ImageNet-Zip/train_map.txt
n01440764/n01440764_10026.JPEG	0
n01440764/n01440764_10027.JPEG	0
n01440764/n01440764_10029.JPEG	0
n01440764/n01440764_10040.JPEG	0
n01440764/n01440764_10042.JPEG	0
```

指定数据集路径：
```bash
export IMAGENET_TRAIN_DATASET=/path/to/dataset
```

### 环境准备
#### 基于base docker image安装
##### 1、导入镜像
```bash
##下载Cambricon PyTorch docker镜像
docker load -i xxx.tar.gz
```

##### 2、启动测试容器
```bash
## run_docker.sh中的path_of_pytorch_modelzoo:path_of_pytorch_modelzoo中，
## 前一个path_of_pytorch_modelzoo为用户host主机端pytorch_modelzoo真实路径，
## 后一个path_of_pytorch_modelzoo为映射到容器内的路径。
## path_of_dataset：path_of_dataset同理。

## 默认的 IMAGE_NAME 已设置为 yellow.hub.cambricon.com/pytorch/pytorch:v1.8.0-torch1.9-ubuntu18.04-py37
## 默认的 MY_CONTAINER 已设置为 swin_transformer_ssl_pytorch_1_9_0

bash run_docker.sh
```

##### 3、在容器中设置环境变量、安装依赖

```bash
## env.sh中的`IMAGENET_TRAIN_DATASET`为容器内imagenet-1k训练数据集的路径，
## 这个环境变量需要用户根据真实情况设置。
## env.sh中的`PTH_AND_LOG_DIR`表示训练过程中保存的pth和log路径，
## 这个环境变量需要用户根据真实情况设置。

source env.sh
pip install -r models/requirements.txt
```

#### 使用Dockerfile准备环境
##### 1、构建 docker 镜像
```bash
export IMAGE_NAME=test_swin_transformer_ssl
## ../../../../路径下包含tools/  built-in/ 等文件夹。
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../
```

##### 2、创建并启动容器(需自行指定数据集目录)

```bash
## 注意：前一个path_of_dataset为用户host主机端数据集存放的路径，
## 后一个path_of_dataset为映射到镜像内的路径。
## 默认的容器名 name 已设置为 test_mlu_swin_transformer_ssl

docker run -it --ipc=host -v path_of_dataset:path_of_dataset -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name test_mlu_swin_transformer_ssl --network=host $IMAGE_NAME
```

##### 3、在容器中设置环境变量、安装依赖

```bash
## env.sh中的`IMAGENET_TRAIN_DATASET`为容器内imagenet-1k训练数据集的路径，
## 这个环境变量需要用户根据真实情况设置。
## env.sh中的`PTH_AND_LOG_DIR`表示训练过程中保存的pth和log路径，
## 这个环境变量需要用户根据真实情况设置。

source env.sh
pip install -r models/requirements.txt
```

### 执行训练或推理脚本
```bash
## 第一阶段训练：Self-Supervised Pre-training
bash run_scripts/Swin_Transformer_SSL_FP32_300E_8MLUs_Pretrain_Train.sh

## 第二阶段训练：Linear Evaluation
## 需要在下述脚本中指定 PRETRAINED_CKPT ，即第一阶段训练得到的某个Epoch权重。
## 该权重可从设置的环境变量PTH_AND_LOG_DIR路径下获取。
bash run_scripts/Swin_Transformer_SSL_FP32_100E_8MLUs_Linear_Train.sh
```
## 一键训练脚本
---
注意：第二阶段训练(Linear Evaluation)，需要在脚本中指定 PRETRAINED_CKPT ，即第一阶段训练得到的某个Epoch权重。该权重可从设置的环境变量 PTH_AND_LOG_DIR 路径下获取。
| Models      | Framework | MLU       | Data Precision | Cards | Description                      | Run                                                         |
| ----------- | --------- | --------- | -------------- | ----- | -------------------------------- | ----------------------------------------------------------- |
| Swin_Transformer_SSL <br> (Self-Supervised Pre-training)  | PyTorch1.9| MLU370-X8 | FP32           | 8     | first stage pretraining use 8 MLUs  | bash run_scripts/Swin_Transformer_SSL_FP32_300E_8MLUs_Pretrain_Train.sh  |
| Swin_Transformer_SSL <br> (Linear Evaluation) | PyTorch1.9| MLU370-X8 | FP32           | 8     | second stage linear evaluation training use 8 MLUs | bash run_scripts/Swin_Transformer_SSL_FP32_100E_8MLUs_Linear_Train.sh |
| Swin_Transformer_SSL <br> (Self-Supervised Pre-training)  | PyTorch1.9| MLU370-X8 | AMP           | 8     | first stage pretraining use 8 MLUs  | bash run_scripts/Swin_Transformer_SSL_AMP_300E_8MLUs_Pretrain_Train.sh  |
| Swin_Transformer_SSL <br> (Linear Evaluation) | PyTorch1.9| MLU370-X8 | AMP           | 8     | second stage linear evaluation training use 8 MLUs | bash run_scripts/Swin_Transformer_SSL_AMP_100E_8MLUs_Linear_Train.sh |


## 一键推理脚本
---
注意: 需要在推理脚本中指定两个权重路径，具体如下:

需要在推理脚本中指定 PRETRAINED_CKPT，这个是第一阶段训练得到的权重，可随意指定某个Epoch的权重。

需要在推理脚本中指定 RESUME_LINEAR， 这个是第二阶段训练得到的权重，也即需要推理测试的权重。

权重可从设置的环境变量PTH_AND_LOG_DIR路径下获取。

| Models      | Framework | MLU       | Data Precision | Description                | Run                                          |
| ----------- | --------- | --------- | -------------- | -------------------------- | -------------------------------------------- |
| Swin_Transformer_SSL | PyTorch1.9  | MLU370-X8 | FP32           | inference script           | bash run_scripts/Swin_Transformer_SSL_Infer.sh |


## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

数据集下载链接：[ImageNet-1k](http://image-net.org/)

## Release_Notes
@TODO
