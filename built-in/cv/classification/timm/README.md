# timm(Pytorch)
---
## 模型概述
  timm系列网络是基于[timm](https://github.com/rwightman/pytorch-image-models)的寒武纪实现版本，采用的timm版本为0.5.0。timm库是一个巨大的PyTorch代码库集合，旨在将各种 SOTA 模型整合在一起，并具有复现 ImageNet 训练结果的能力。

## 支持情况
---
### 训练模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-MLUs |
----- | ----- | ----- | ----- | ----- |
Inception_v4  | PyTorch1.9  | MLU370-X8  | AMP/FP32  | Yes  |
Inception_v3  | PyTorch1.6  | MLU370-X8  | AMP/FP32  | Yes  |

### 推理模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Supported Infer Mode | 
----- | ----- | ----- | ----- | ----- |
Inception_v4  | PyTorch1.9  | MLU370-X8  | FP32      | CNNL |
Inception_v3  | PyTorch1.6  | MLU370-X8  | FP32      | CNNL |

## 默认参数配置
---
### Optimizer
Models  | 优化器  | Learning Rate   | Learning rate schedule |  Weight decay | Epoch
---- | ----- | ----- | ----- | ----- | ----- |
Inception_v4  | Rmsproptf  | 0.0224  | Step  | 1e-5 | 300 |
Inception_v3  | sgd  | 0.045  | cosine  | 4e-5 | 200 |

### Data Augmentation
模型使用了以下数据增强方法：

训练
* RandomResizedCropAndInterpolation
* RandomHorizontalFlip
* RandAugment

推理
* Resize
* CenterCrop

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
- `train.py` 模型训练入口，更多信息使用`python train.py -h`查看
- `validate.py` 模型推理入口，更多信息使用`python validate.py -h`查看

### 准备数据集
下载数据集[ImageNet-2012](http://image-net.org/)，并解压。解压后的数据集请放在` $IMAGENET_TRAIN_DATASET`目录下(`IMAGENET_TRAIN_DATASET`是需要设置的环境变量)，目录结构为：
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

## 默认的 IMAGE_NAME 已设置为 yellow.hub.cambricon.com/pytorch/pytorch:v1.8.0-torch1.9-ubuntu18.04-py37，这是针对Inception_v4网络的
## 如果需要运行Inception_v3网络，则需要把IMAGE_NAME设置为：yellow.hub.cambricon.com/pytorch/pytorch:v1.8.0-torch1.6-ubuntu18.04-py36或者
## yellow.hub.cambricon.com/pytorch/pytorch:v1.8.0-torch1.6-ubuntu18.04-py37等torch1.6的镜像。
## 默认的 MY_CONTAINER 已设置为 timm_pytorch_1_9_0，用户可根据实际情况修改。

bash run_docker.sh
```

##### 3、在容器中设置环境变量、安装依赖

```bash
## env.sh中的`IMAGENET_TRAIN_DATASET`为容器内imagenet-2012训练数据集的路径，
## 这个环境变量需要用户根据真实情况设置。
## env.sh中的`PTH_AND_LOG_DIR`表示训练过程中保存的pth和log路径，
## 这个环境变量需要用户根据真实情况设置。

source env.sh
pip install -r models/requirements.txt
```

#### 使用Dockerfile准备环境
##### 1、构建 docker 镜像
```bash
##IMAGE_NAME用户可根据实际情况修改。
export IMAGE_NAME=test_timm_pytorch_1_9
## ../../../../路径下包含tools/  built-in/ 等文件夹。
## DOCKERFILE中 FROM_IMAGE_NAME默认设置为yellow.hub.cambricon.com/pytorch/pytorch:v1.8.0-torch1.9-ubuntu18.04-py37，这是针对Inception_v4网络的
## 如果需要运行Inception_v3网络，则需要把FROM_IMAGE_NAME设置为：yellow.hub.cambricon.com/pytorch/pytorch:v1.8.0-torch1.6-ubuntu18.04-py36或者
## yellow.hub.cambricon.com/pytorch/pytorch:v1.8.0-torch1.6-ubuntu18.04-py37等torch1.6的镜像。

docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../
```

##### 2、创建并启动容器(需自行指定数据集目录)

```bash
## 注意：前一个path_of_dataset为用户host主机端数据集存放的路径，
## 后一个path_of_dataset为映射到镜像内的路径。
## 默认的容器名 name 已设置为 test_mlu_timm

docker run -it --ipc=host -v path_of_dataset:path_of_dataset -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name test_mlu_timm --network=host $IMAGE_NAME
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
bash run_scripts/Inception_v4/Inceptionv4_FP32_300E_8MLUs_Train.sh
```

## 一键训练脚本
---
注意：训练过程中，生成的权重和log保存在设置的环境变量 PTH_AND_LOG_DIR 路径下。
| Models      | Framework | MLU       | Data Precision | Cards | Description                      | Run                                                         |
| ----------- | --------- | --------- | -------------- | ----- | -------------------------------- | ----------------------------------------------------------- |
| Inception_v4 | PyTorch1.9 | MLU370-X8 | FP32          | 8     | from scratch training use 8 MLUs  | bash run_scripts/Inception_v4/Inceptionv4_FP32_300E_8MLUs_Train.sh  |
| Inception_v4 | PyTorch1.9 | MLU370-X8 | AMP           | 8     | from scratch training use 8 MLUs  | bash run_scripts/Inception_v4/Inceptionv4_AMP_300E_8MLUs_Train.sh   |
| Inception_v3 | PyTorch1.6 | MLU370-X8 | FP32          | 4     | from scratch training use 4 MLUs  | bash run_scripts/Inception_v3/Inceptionv3_FP32_200E_4MLUs_Train.sh  |
| Inception_v3 | PyTorch1.6 | MLU370-X8 | AMP           | 4     | from scratch training use 4 MLUs  | bash run_scripts/Inception_v3/Inceptionv3_AMP_200E_4MLUs_Train.sh   |

## 一键推理脚本
---
注意: 需要在推理脚本中指定 EVAL_CKPT，这个是需要推理测试的权重。

权重可从设置的环境变量 PTH_AND_LOG_DIR 路径下获取。

| Models      | Framework | MLU       | Data Precision | Description                | Run                                          |
| ----------- | --------- | --------- | -------------- | -------------------------- | -------------------------------------------- |
| Inception_v4 | PyTorch1.9  | MLU370-X8 | FP32           | inference script           | bash run_scripts/Inception_v4/Inceptionv4_Infer.sh |
| Inception_v3 | PyTorch1.6  | MLU370-X8 | FP32           | inference script           | bash run_scripts/Inception_v3/Inceptionv3_Infer.sh |


## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

数据集下载链接：[ImageNet-2012](http://image-net.org/)

## Release_Notes
@TODO

