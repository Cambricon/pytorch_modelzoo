# vision_classification(PyTorch)
## 模型概述
vision_classification系列网络是基于 [torchvision.models(v0.7.0)](https://github.com/pytorch/vision/tree/v0.7.0/torchvision/models)的寒武纪实现版本，torchvision是pytorch用来处理图像，构建计算机视觉模型的算法库，其中torchvision.models包含常用的算法模型结构，且含预训练模型，例如ResNet50、VGG16、AlexNet等；

## 支持情况
### 训练模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-MLUS |
----- | ----- | ----- | ----- | ----- |
ResNet50  | PyTorch1.6 | MLU370-X8  | AMP/FP32  | Yes  |
ResNet18  | PyTorch1.6 | MLU370-X8  | FP32  | Yes |
VGG16  | PyTorch1.6 | MLU370-X8  | AMP/FP32  | Yes | 
MobileNetv2  | PyTorch1.6 | MLU370-X8  | AMP/FP32  | Yes |
AlexNet  | PyTorch1.6 | MLU370-X8  | FP32  | Yes |  
GoogleNet  | PyTorch1.6 | MLU370-X8  | AMP/FP32  | Yes | 
ResNet101  | PyTorch1.6 | MLU370-X8  | FP32  | Yes |
VGG19  | PyTorch1.6 | MLU370-X8  | FP32  | Yes |
VGG16_bn  | PyTorch1.6 | MLU370-X8  | FP32  | Yes | 
ShuffleNet_v2_x0_5  | PyTorch1.6 | MLU370-X8  | FP32  | Yes |
ShuffleNet_v2_x1_0  | PyTorch1.6 | MLU370-X8  | FP32  | Yes | 
ShuffleNet_v2_x1_5  | PyTorch1.6 | MLU370-X8  | FP32  | Yes |

### 推理模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision   | Supported Infer Mode | 
----- | ----- | ----- | ----- | ----- |
ResNet50  | PyTorch1.6 | MLU370-S4/X4  | FP16/FP32  | TorchMM/CNNL | 
ResNet18  | PyTorch1.6 | MLU370-S4/X4  | FP16/FP32  | TorchMM/CNNL | 
VGG16  | PyTorch1.6 | MLU370-S4/X4  | FP16/FP32  | TorchMM/CNNL | 
MobileNetv2  | PyTorch1.6 | MLU370-S4/X4  | FP16/FP32  | TorchMM/CNNL |
AlexNet  | PyTorch1.6 | MLU370-S4/X4  | FP16/FP32  | TorchMM/CNNL |
GoogleNet  | PyTorch1.6 | MLU370-S4/X4  | FP16/FP32  | TorchMM/CNNL |
ResNet101  | PyTorch1.6 | MLU370-S4/X4  | FP16/FP32  | TorchMM/CNNL |
VGG19  | PyTorch1.6 | MLU370-S4/X4  | FP16/FP32  | TorchMM/CNNL |
VGG16_bn  | PyTorch1.6 | MLU370-S4/X4  | FP16/FP32  | TorchMM/CNNL |
ShuffleNet_v2_x0_5  | PyTorch1.6 | MLU370-S4/X4  | FP16/FP32  | TorchMM/CNNL |
ShuffleNet_v2_x1_0  | PyTorch1.6 | MLU370-S4/X4  | FP16/FP32  | TorchMM/CNNL |
ShuffleNet_v2_x1_5  | PyTorch1.6 | MLU370-S4/X4  | FP16/FP32  | TorchMM/CNNL |


## 默认参数配置
### 模型训练默认参数配置
以下为vision_classification模型的默认参数配置：

### Optimizer
Models  | 优化器  | Learning Rate   | Learning rate schedule |  Weight decay | Label Smoothing | Epoch
---- | ----- | ----- | ----- | ----- | ----- |---- |
ResNet50  | SGD  | 0.1  | Linear schedule  | 1e-4 | None | 100
ResNet18  | SGD  | 0.2  | Linear schedule  | 1e-4 | None | 150
VGG16  | SGD  | 0.02  | Linear schedule  | 1e-4 | None | 100
MobileNetv2  | SGD  | 0.05  | Linear schedule  | 4e-5 | None | 100
AlexNet  | SGD  | 0.04  | Linear schedule  | 1e-4 | None | 150
GoogleNet  | SGD  | 0.1  | Linear schedule  | 4e-5 | None | 150
ResNet101  | SGD  | 0.1  | Linear schedule  | 4e-5 | None | 100
VGG19  | SGD  | 0.01  | Linear schedule  | 1e-4 | None | 100
VGG16_bn  | SGD  | 0.01  | Linear schedule  | 1e-4 | None | 150
ShuffleNet_v2_x0_5  | SGD  | 0.5  | Linear schedule  | 4e-5 | None | 300
ShuffleNet_v2_x1_0  | SGD  | 0.5  | Linear schedule  | 4e-5 | None | 300
ShuffleNet_v2_x1_5  | SGD  | 0.25  | Linear schedule  | 4e-5 | None | 300

### Data Augmentation
模型训练使用了以下数据增强方法： 

* Normolization
* Crop image to 224*224
* RandomHorizontalFlip

### 模型推理默认参数配置
* ckpt: 没有指定该参数的情况下，默认会从$TORCH_HOME/checkpoints/下加载模型，找不到模型的情况下，还可以自动从torchvision官网下载，也可选指定训练完成的权重(eg. --ckpt /model/xxxx.pt)
* batch_size: 64
* input_data_type：默认使用 float32
* qint：默认不开启量化


## 环境依赖
* Linux常见操作系统版本(如Ubuntu18.04，Ubuntu20.04, CentOS7.6)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算板卡MLU370-X8;
* Cambricon Driver >=v4.20.6；
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 快速入门指南

### 文件说明
- run_scripts/ 包含一键训练和推理的shell脚本文件
- models/ 包含原始模型仓库文件
- `classify_train.py` 模型训练脚本，更多信息使用`python classify_train.py -h`查看
- `classify_infer.py` 模型推理验证脚本, 更多信息使用`python classify_infer.py -h`查看

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
#path_of_datase：path_of_dataset同理。

bash run_docker.sh
```

##### 3、启动虚拟环境，安装依赖，并设置环境变量

```
#1、env.sh中的`IMAGENET_TRAIN_DATASET`为容器内imagenet训练数据集的路径，这个环境变量需要用户根据真实情况设置。 
#2、TORCH_HOME为torchvision自动下载模型时保存的路径，用户可以根据实际情况设置，DATASET_NAME描述本Demo使用的数据集名称。 
#3、env.sh中的`IMAGENET_INFER_CHECKPOINT`为ShuffleNet_v2_x1_5模型使用--ckpt参数加载模型checkpoint的路径，本工程只有ShuffleNet_v2_x1_5提供了ckpt例子。

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
#注意：前一个path_of_dataset为用户host主机端数据集存放的路径，后一个path_of_dataset为映射到镜像内的路径。

docker run -it --ipc=host -v path_of_dataset:path_of_dataset -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name test_classify --network=host $IMAGE_NAME
```

##### 3、启动虚拟环境，设置环境变量

```
#1、env.sh中的`IMAGENET_TRAIN_DATASET`为容器内imagenet训练数据集的路径，这个环境变量需要用户根据真实情况设置。 
#2、TORCH_HOME为torchvision自动下载模型时保存的路径，用户可以根据实际情况设置，DATASET_NAME描述本Demo使用的数据集名称。
#3、env.sh中的`IMAGENET_INFER_CHECKPOINT`为ShuffleNet_v2_x1_5模型使用--ckpt参数加载模型checkpoint的路径，本工程只有ShuffleNet_v2_x1_5提供了ckpt例子。 

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
bash run_scripts/ResNet50/ResNet50_AMP_100E_4MLUs_Train.sh
```

#### 一键执行训练脚本
Models  | Framework  | MLU   | MODE  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50  | PyTorch1.6 | MLU370-X8  |  AMP(from scratch)  | 4  | bash run_scripts/ResNet50/ResNet50_AMP_100E_4MLUs_Train.sh
ResNet50  | PyTorch1.6 | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/ResNet50/ResNet50_FP32_100E_4MLUs_Train.sh
ResNet18  | PyTorch1.6 | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/ResNet18/ResNet18_FP32_100E_4MLUs_Train.sh
VGG16  | PyTorch1.6 | MLU370-X8  |  AMP(from scratch)  | 4  | bash run_scripts/VGG16/VGG16_AMP_100E_4MLUs_Train.sh
VGG16  | PyTorch1.6 | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/VGG16/VGG16_FP32_100E_4MLUs_Train.sh
MobileNet_v2  | PyTorch1.6 | MLU370-X8  |  AMP(from scratch)  | 4  | bash run_scripts/MobileNet_v2/MobileNetv2_AMP_150E_4MLUs_Train.sh
MobileNet_v2  | PyTorch1.6 | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/MobileNet_v2/MobileNetv2_FP32_150E_4MLUs_Train.sh
AlexNet  | PyTorch1.6 | MLU370-X8  | FP32(from scratch)  | 8  | bash run_scripts/AlexNet/AlexNet_FP32_100E_8MLUs_Train.sh
GoogleNet  | PyTorch1.6 | MLU370-X8  |  AMP(from scratch)  | 4  | bash run_scripts/GoogleNet/GoogleNet_AMP_150E_4MLUs_Train.sh
GoogleNet  | PyTorch1.6 | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/GoogleNet/GoogleNet_FP32_150E_4MLUs_Train.sh
ResNet101  | PyTorch1.6 | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/ResNet101/ResNet101_FP32_100E_4MLUs_Train.sh
VGG19  | PyTorch1.6 | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/VGG19/VGG19_FP32_100E_4MLUs_Train.sh
VGG16_bn  | PyTorch1.6 | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/VGG16_bn/VGG16_bn_FP32_100E_4MLUs_Train.sh
ShuffleNet_v2_x0_5  | PyTorch1.6 | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/ShuffleNet_v2_x0_5/ShuffleNetv2x05_FP32_300E_4MLUs_Train.sh
ShuffleNet_v2_x1_0  | PyTorch1.6 | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/ShuffleNet_v2_x1_0/ShuffleNetv2x10_FP32_300E_4MLUs_Train.sh
ShuffleNet_v2_x1_5  | PyTorch1.6 | MLU370-X8  | FP32(from scratch)  | 4  | bash run_scripts/ShuffleNet_v2_x1_5/ShuffleNetv2x15_FP32_300E_4MLUs_Train.sh


#### 一键执行推理脚本

> 注意：ShuffleNetv2x15_Infer.sh脚本中，由于torchvision官方没有提供预训练模型，所以无法自动下载模型推理，建议通过设置env.sh中的`IMAGENET_INFER_CHECKPOINT`进行推理。

Models  | Framework  | MLU   |Run
----- | ----- | ----- | ----- | 
ResNet50  | PyTorch1.6 | MLU370-S4/MLU370-X4 | bash run_scripts/ResNet50/ResNet50_Infer.sh
ResNet18  | PyTorch1.6 | MLU370-S4/MLU370-X4 | bash run_scripts/ResNet18/ResNet18_Infer.sh
VGG16  | PyTorch1.6 | MLU370-S4/MLU370-X4 | bash run_scripts/VGG16/VGG16_Infer.sh
MobileNetv2  | PyTorch1.6 | MLU370-S4/MLU370-X4 | bash run_scripts/MobileNet_v2/MobileNetv2_Infer.sh
AlexNet  | PyTorch1.6 | MLU370-S4/MLU370-X4 | bash run_scripts/AlexNet/AlexNet_Infer.sh
GoogleNet  | PyTorch1.6 | MLU370-S4/MLU370-X4 | bash run_scripts/GoogleNet/GoogleNet_Infer.sh
ResNet101  | PyTorch1.6 | MLU370-S4/MLU370-X4 | bash run_scripts/ResNet101/ResNet101_Infer.sh
VGG19  | PyTorch1.6 | MLU370-S4/MLU370-X4 | bash run_scripts/VGG19/VGG19_Infer.sh
VGG16_bn  | PyTorch1.6 | MLU370-S4/MLU370-X4 | bash run_scripts/VGG16_bn/VGG16_bn_Infer.sh
ShuffleNet_v2_x0_5  | PyTorch1.6 | MLU370-S4/MLU370-X4 | bash run_scripts/ShuffleNet_v2_x0_5/ShuffleNetv2x05_Infer.sh
ShuffleNet_v2_x1_0  | PyTorch1.6 | MLU370-S4/MLU370-X4 | bash run_scripts/ShuffleNet_v2_x1_0/ShuffleNetv2x10_Infer.sh
ShuffleNet_v2_x1_5  | PyTorch1.6 | MLU370-S4/MLU370-X4 | bash run_scripts/ShuffleNet_v2_x1_5/ShuffleNetv2x15_Infer.sh




## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

ImageNet1K 数据集下载链接：https://www.image-net.org/       \
torchvision.models 模型代码链接：https://github.com/pytorch/vision/tree/v0.7.0/torchvision/models


## Release_Notes
@TODO
