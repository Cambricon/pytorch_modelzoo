# ngc-resnet50v1_5(Pytorch)
---
## 模型概述
本仓库为ngc-resnet50v1.5的MLU实现，GPU实现可参考仓库:[resnet50v1.5](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/)

## 支持情况
### 训练模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-MLUS |
----- | ----- | ----- | ----- | ----- |
ngc-resnet50v1.5  | PyTorch1.6  | MLU370-X8  | AMP/FP32  | Yes  |

### 推理模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision   | Supported Infer Mode | 
----- | ----- | ----- | ----- | ----- |
ngc-resnet50v1.5  | PyTorch1.6  | MLU370-S4/X4  | FP16/FP32  | CNNL |

## 默认参数配置
以下为ngc-resnet50v1.5模型的默认参数配置：

### Optimizer
Models  | 优化器  | Learning Rate   | Learning rate schedule |  Weight decay | Label Smoothing | Epoch
---- | ----- | ----- | ----- | ----- | ----- |---- |
ngc-resnet50v1.5  | SGD  | 2.048  | step schedule  | 1e-4 | None | 90

### Data Augmentation
模型使用了以下数据增强方法：
* 训练
    * RandomSizedCrop
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
- `models/main.py` 模型训练&推理入口，更多信息使用`python main.py -h`查看
- `models/multiproc.py` 分布式处理入口，更多信息使用`python multiproc.py -h`查看

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
#1、env.sh中的`IMAGENET_TRAIN_DATASET`为容器内ImageNet2012训练数据集的路径，这个环境变量需要用户根据真实情况设置。 

source env.sh
```


#### 使用Dockerfile准备环境
#### 1、生成ngc-resnet50v1.5的Docker镜像：

```
export IMAGE_NAME=test_ngc_resnet50v1.5
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../
```

####  2、创建容器

```
#注意：前一个path_of_dataset为用户host主机端数据集存放的路径，后一个path_of_dataset为映射到镜像内的路径。

docker run -it --ipc=host -v path_of_dataset:path_of_dataset -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name test_ngc_resnet50v1.5_classify --network=host $IMAGE_NAME
```

##### 3、启动虚拟环境，设置环境变量

```
#1、env.sh中的`IMAGENET_TRAIN_DATASET`为容器内ImageNet2012训练数据集的路径，这个环境变量需要用户根据真实情况设置。 

source /torch/venv3/pytorch/bin/activate
source env.sh
```

### 数据集准备
该ngc_resnet50v1.5系列模型基于ILSVRC2012数据集训练，下载链接：<https://www.image-net.org/>。数据集请放在` $IMAGENET_TRAIN_DATASET`目录下。目录结构为：
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
bash run_scripts/NGC_ResNet50v15_AMP_90E_4MLUS_Train.sh
```

#### 一键执行训练脚本

> Attention: checkpoint以及log都输出在models/下面，checkpoint包括model_best.pth.tar和checkpoint.pth.tar,log包括raport.json，GPU_*.log。

Models  | Framework  | MLU   | MODE  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
ngc-resnet50v1_5  | PyTorch  | MLU370-X8  |  AMP(from scratch)  | 4  | bash run_scripts/NGC_ResNet50v15_AMP_90E_4MLUS_Train.sh
ngc-resnet50v1_5  | PyTorch  | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/NGC_ResNet50v15_FP32_90E_4MLUS_Train.sh


#### 一键执行推理脚本

> Attention: 执行推理脚本时，首先要确保--resume后的checkpoint路径存在，且--epochs的数值必须为checkpoint的epoch数值加一。

Models  | Framework  | MLU   |Run
----- | ----- | ----- | ----- | 
ngc-resnet50v1_5  | PyTorch  | MLU370-S4/MLU370-X4 | bash run_scripts/NGC_ResNet50v15_Infer.sh


## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

ImageNet2012 数据集下载链接：https://www.image-net.org/       \


## Release_Notes
@TODO

          

