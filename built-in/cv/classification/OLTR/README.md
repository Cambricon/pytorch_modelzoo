# OLTR(Pytorch)
---
## 模型概述
  OLTR网络源于论文[Large-Scale Long-Tailed Recognition in an Open World](https://arxiv.org/abs/1904.05160)

  本仓库为OLTR的MLU实现，GPU实现可参考仓库: [OpenLongTailRecognition-OLTR](https://github.com/naviocean/pseudo-3d-pytorch/tree/50297d11248630792709782f467982e80c281384)

## 支持情况
### 训练模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-MLUS |
----- | ----- | ----- | ----- | ----- |
OLTR  | PyTorch1.6  | MLU370-X8  | FP32  | Yes  |

### 推理模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision   | Supported Infer Mode | 
----- | ----- | ----- | ----- | ----- |
OLTR  | PyTorch1.6  | MLU370-S4/X4  | FP16/FP32  | CNNL |

## 默认参数配置
以下为OLTR模型的默认参数配置：

### Stage1
Model Part  | 优化器  | Learning Rate   | Learning rate schedule |  Weight decay | momentum |
---- | ----- | ----- | ----- | ----- | ----- |
feature  | SGD  | 1e-3  | StepLR | 5e-4 | 9e-1 | 
classifier  | SGD  | 1e-3  | StepLR | 5e-4 | 9e-1 |

### Stage2
Model Part  | 优化器  | Learning Rate   | Learning rate schedule |  Weight decay | momentum |
---- | ----- | ----- | ----- | ----- | ----- |
feature  | SGD  | 1e-2  | StepLR | 5e-4 | 9e-1 |
classifier  | SGD  | 1e-1  | StepLR | 5e-4 | 9e-1 |
feat_loss  | SGD  | 1e-4  | StepLR | 5e-4 | 9e-1 |

### Data Augmentation
模型使用了以下数据增强方法：
* 训练
    * RandomResizedCrop to 224
    * RandomHorizontalFlip
    * ColorJitter
    * Normolization
* 验证
    * Resize to 256
    * CenterCrop to 224   
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
  - `main_imagenet.py` 模型训练&推理入口，更多信息使用`python main_imagenet.py -h`查看
  - `run_networks.py` 模型训练脚本


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
#1、env.sh中的`PYTORCH_TRAIN_DATASET`为容器内ImageNet_LT训练数据集路径，这个环境变量需要用户根据真实情况设置。 
#2、env.sh中的`IMAGENET_TRAIN_DATASET`为容器内ImageNet训练数据集路径，这个环境变量需要用户根据真实情况设置。 

source env.sh
pip install -r models/requirements.txt
```


#### 使用Dockerfile准备环境
#### 1、生成vision_classification的Docker镜像：

```
export IMAGE_NAME=test_oltr
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../
```

####  2、创建容器

```
#注意：前一个path_of_dataset为用户host主机端数据集存放的路径，后一个path_of_dataset为映射到镜像内的路径。

docker run -it --ipc=host -v path_of_dataset:path_of_dataset -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name test_oltr_classify --network=host $IMAGE_NAME
```

##### 3、启动虚拟环境，设置环境变量

```
#1、env.sh中的`PYTORCH_TRAIN_DATASET`为容器内ImageNet_LT训练数据集路径，这个环境变量需要用户根据真实情况设置。 
#2、env.sh中的`IMAGENET_TRAIN_DATASET`为容器内ImageNet训练数据集路径，这个环境变量需要用户根据真实情况设置。  

source env.sh
```

### 数据集准备
首先，请下载 ImageNet_2014 数据集，下载链接：<https://image-net.org/index>，并将其放在`$IMAGENET_TRAIN_DATASET`目录下，目录结构为：
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

接下来，请从这里下载 ImageNet-LT数据集，下载链接：<https://drive.google.com/drive/folders/19cl6GK5B3p5CxzVBy5i4cWSmBy9-rT_->，并将其放在`$PYTORCH_TRAIN_DATASET`目录下，目录结构为：
```
├── ImageNet_LT
│   ├── ILSVRC2010_val_00000007.JPEG
│   ├── ILSVRC2010_val_00000010.JPEG
│   ├── ILSVRC2010_val_00000012.JPEG
│   ├── ILSVRC2010_val_00000015.JPEG
│   ├── ILSVRC2010_val_00000016.JPEG
│   ├── ...
│   ├── open.txt
│   ├── test.txt
│   ├── train.txt
│   ├── val.txt
```

### Run 脚本执行
```
bash run_scripts/OLTR_FP32_1MLUs_Train.sh
```

#### 一键执行训练脚本

> Attention: 脚本OLTR_FP32_1MLUs_Train.sh中，Stage1训练出来的模型保存在 models/logs/ImageNet_LT/imagenet_mid_stage1/final_model_checkpoint.pth路径下， \
Stage2训练出来的模型保存在models/logs/ImageNet_LT/imagenet_mid_meta_embedding_exp2/final_model_checkpoint.pth路径下。

Models  | Framework  | MLU   | MODE  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
OLTR  | PyTorch  | MLU370-X8  | FP32(from scratch)  | 1  | bash run_scripts/OLTR_FP32_1MLUs_Train.sh


#### 一键执行推理脚本

> Attention: 脚本OLTR_Infer.sh中，必须确保models/logs/ImageNet_LT/imagenet_mid_meta_embedding_exp2/final_model_checkpoint.pth路径存在才能成功推理。

Models  | Framework  | MLU   |Run
----- | ----- | ----- | ----- | 
OLTR  | PyTorch  | MLU370-S4/MLU370-X4 | bash run_scripts/OLTR_Infer.sh


## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

ImageNet-LT 数据集下载链接：https://drive.google.com/drive/folders/19cl6GK5B3p5CxzVBy5i4cWSmBy9-rT_-      \
ImageNet_2014数据集下载链接：https://image-net.org/index


## Release_Notes
@TODO
