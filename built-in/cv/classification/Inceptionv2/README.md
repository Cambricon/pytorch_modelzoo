# Inceptionv2(Pytorch)
---
## 模型概述
  Inceptionv2网络源于论文[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)

  本仓库为Inceptionv2的MLU实现，GPU实现可参考仓库: [bninception](https://github.com/Cadene/pretrained-models.pytorch/blob/1e1bca074f29db9e10c18ca8b51e031c89d4446a/pretrainedmodels/models/bninception.py)

## 支持情况
### 训练模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-MLUS |
----- | ----- | ----- | ----- | ----- |
Inceptionv2  | PyTorch1.6  | MLU370-X8  | FP32/AMP  | Yes  |

### 推理模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision   | Supported Infer Mode | 
----- | ----- | ----- | ----- | ----- |
Inceptionv2  | PyTorch1.6  | MLU370-S4/X4  | FP16/FP32  | CNNL |

## 默认参数配置
以下为Inceptionv2模型的默认参数配置：

Models  | 优化器  | Learning Rate   | Learning rate schedule |  Weight decay | momentum |
---- | ----- | ----- | ----- | ----- | ----- |
Inceptionv2  | SGD  | 1e-2  | StepLR | 1e-4 | 9e-1 | 


### Data Augmentation
模型使用了以下数据增强方法：
* 训练
    * RandomResizedCrop
    * RandomHorizontalFlip
    * Normolization
* 验证
    * Resize
    * CenterCrop
    * Normolization


## 环境依赖
* Linux常见操作系统版本(如Ubuntu18.04，Ubuntu20.04, CentOS7.6)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算板卡MLU370-X8;
* Cambricon Driver >=v4.20.6；
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 快速入门指南

### 文件说明
- run_scripts/ 包含一键训练和推理的shell脚本文件
- models/ 包含原始模型仓库文件
  - `src/train.py` 模型训练&推理入口，更多信息使用`python train.py -h`查看
  - `src/model/` 网络架构代码。


### 环境准备
#### 基于base docker image安装
##### 1、导入镜像
```
##下载Cambricon PyTorch docker镜像
docker load -i xxx.tar.gz
```

##### 2、启动测试容器
```
#run_docker.sh中的path_of_pytorch_modelzoo:path_of_pytorch_modelzoo中，
#前一个path_of_pytorch_modelzoo为用户host主机端pytorch_modelzoo真实路径，
#后一个path_of_pytorch_modelzoo为映射到容器内的路径。
#path_of_dataset：path_of_dataset同理。

bash run_docker.sh
```

##### 3、启动虚拟环境，安装依赖，并设置环境变量

```
#1、env.sh中的`IMAGENET_TRAIN_DATASET`为容器内ImageNet训练数据集路径，这个环境变量需要用户根据真实情况设置。 

source env.sh
pip install -r models/requirements.txt
```


#### 使用Dockerfile准备环境
#### 1、生成Inceptionv2的Docker镜像：

```
export IMAGE_NAME=test_inceptionv2
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../
```

####  2、创建容器

```
#注意：前一个path_of_dataset为用户host主机端数据集存放的路径，后一个path_of_dataset为映射到镜像内的路径。

docker run -it --ipc=host -v path_of_dataset:path_of_dataset -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name test_inceptionv2_classify --network=host $IMAGE_NAME
```

##### 3、启动虚拟环境，设置环境变量

```
#1、env.sh中的`IMAGENET_TRAIN_DATASET`为容器内ImageNet训练数据集路径，这个环境变量需要用户根据真实情况设置。

source env.sh
```

### 数据集准备
该vision_classification系列模型基于ILSVRC2012数据集训练，下载链接：<https://www.image-net.org/>。数据集请放在` $IMAGENET_TRAIN_DATASET`目录下。目录结构为：
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

### Run 脚本执行
```
bash run_scripts/Inceptionv2_FP32_100E_8MLUs_Train.sh
```

#### 一键执行训练脚本

> Attention: 训练出来的checkpoints保存在 `models/logs_inception_v2` 路径下， \
训练出来的日志保存在`models/logs_inception_v2`和 `models/iters_inception_v2`(仅限于第一个epoch)路径下。

Models  | Framework  | MLU   | MODE  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
Inceptionv2  | PyTorch  | MLU370-X8  | AMP(from scratch)  | 8 | bash run_scripts/Inceptionv2_AMP_100E_8MLUs_Train.sh |
Inceptionv2  | PyTorch  | MLU370-X8  | FP32(from scratch)  | 8 | bash run_scripts/Inceptionv2_FP32_100E_8MLUs_Train.sh |



## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

ImageNet_2012数据集下载链接：https://www.image-net.org/


## Release_Notes
@TODO
