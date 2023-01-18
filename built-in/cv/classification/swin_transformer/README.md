# TIMM-Swin-transformer
---
## 模型概述
  本仓库是基于[pytorch-image-models](https://github.com/rwightman/pytorch-image-models)的swin-transformer MLU实现

## 支持情况
### 训练模型支持情况
| Models                | Framework  | Supported MLU | Supported Data Precision | Multi-MLUS |
| --------------------- | ---------- | ------------- | ------------------------ | ---------- |
| Timm-swin-transformer | PyTorch1.6 | MLU370-X8     | AMP                      | Yes        |

### 推理模型支持情况
| Models                | Framework  | Supported MLU | Supported Data Precision | Supported Infer Mode |
| --------------------- | ---------- | ------------- | ------------------------ | -------------------- |
| Timm-swin-transformer | PyTorch1.6 | MLU370-S4/X4  | FP16/FP32                | CNNL                 |

## 默认参数配置
### 模型训练默认参数

#### Optimizer

模型优化器为Adamw，参数配置如下：

- Learning Rate: 1e-3 for batch size 128
- beta =(0.9, 0.999)
- epsilon=1e-8
- weight_decay: 0.05
- Epoch:300


## 环境依赖
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算板卡MLU370-X8;
* Cambricon Driver >=v4.20.6；
* CNToolKit >=2.8.3;
* CNNL >=1.10.2;
* CNCL >=1.1.1;
* CNLight >=0.12.0;
* CNPyTorch >= 1.3.0;
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 快速入门指南

### 文件说明
- run_scripts/ 包含一键训练和推理的shell脚本文件
- models/ 包含原始模型仓库文件
  - `train.py` 模型训练入口，更多信息使用`python train.py -h`查看
  - `validate.py` 模型推理入口，更多信息使用`python validate.py -h`查看

### 数据集准备

下载 ImageNet2012 数据集，下载链接：<https://image-net.org/index>，并将其放在`$IMAGENET_TRAIN_DATASET`目录下，目录结构为：

```
├── train
│   ├── n01440764
│   ├── n01443537
│   ├── ...
├── train.txt
├── val
│   ├── n01440764
│   ├── n01443537
│   ├── ...
└── val.txt
```

### 环境准备
#### 基于base docker image安装
##### 1、导入镜像
```
##下载Cambricon PyTorch docker镜像
docker load -i xxx.tar.gz
```

##### 2、启动测试容器（根据下载的镜像名修改脚本）
```
#修改run_docker.sh中的/your/data:/your/data，其中
#前一个/your/data为用户host主机端data真实路径，
#后一个/your/data为映射到容器内的路径。
bash run_docker.sh
```

##### 3、启动虚拟环境，安装依赖，并设置环境变量

```
#1、env.sh中的`IMAGENET_TRAIN_DATASET`为容器内ImageNet2012训练数据集路径，这个环境变量需要用户根据真实情况设置。 
#2、env.sh中的`TIMM_INFER_MODEL`为容器内Timm-swin-transformer 推理的模型文件，这个环境变量需要用户根据真实情况设置。 
source env.sh
source /torch/venv3/pytorch/bin/activate
pip install -r models/requirements.txt
```


#### 使用Dockerfile准备环境
1、构建Docker镜像：

```
export IMAGE_NAME=test_timm
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../
```

2、创建容器

```
#注意：前一个path_of_dataset为用户host主机端数据集存放的路径，后一个path_of_dataset为映射到镜像内的路径。

docker run -it --ipc=host -v path_of_dataset:path_of_dataset -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name test_timm_swintransformer --network=host $IMAGE_NAME
```

3、启动虚拟环境，设置环境变量

```
#1、env.sh中的`IMAGENET_TRAIN_DATASET`为容器内ImageNet2012训练数据集路径，这个环境变量需要用户根据真实情况设置。 
#2、env.sh中的`TIMM_INFER_MODEL`为容器内Timm-swin-transformer 推理的模型文件，这个环境变量需要用户根据真实情况设置。 
source env.sh
source /torch/venv3/pytorch/bin/activate
pip install -r models/requirements.txt
```

### Run 脚本执行
```
bash run_scripts/timm_swin_transform_AMP_300E_8MLUs_Train.sh
```

#### 一键执行训练脚本

> 训练出来的模型保存在models/SwinTransformer/tmp/*/model_best.pth.tar路径下。其中\*表示一个按时间记录的文件夹，例如20221212-195803-swin_tiny_patch4_window7_224-224

| Models                | Framework | MLU       | MODE              | Cards | Run                                                          |
| --------------------- | --------- | --------- | ----------------- | ----- | ------------------------------------------------------------ |
| Timm-swin-transformer | PyTorch   | MLU370-X8 | AMP(from scratch) | 8     | bash run_scripts/timm_swin_transformer_AMP_300E_8MLUs_Train.sh |


#### 一键执行推理脚本

> 在推理前需要设置TIMM_INFER_MODEL环境变量，指定需要推理的模型，如果是经过上面训练脚步执行后的，模型位置一般在models/SwinTransformer/tmp/*/model_best.pth.tar,其中\*表示一个按时间记录的文件夹，例如20221212-195803-swin_tiny_patch4_window7_224-224

| Models                | Framework | MLU                 | Run                                             |
| --------------------- | --------- | ------------------- | ----------------------------------------------- |
| Timm-swin-transformer | PyTorch   | MLU370-S4/MLU370-X4 | bash run_scripts/timm_swin_transformer_Infer.sh |


## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

ImageNet2012数据集下载链接：https://image-net.org/index


## Release_Notes
@TODO
