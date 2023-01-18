# YOLOV3(PyTorch)
## 模型概述

该sample包含PYTORCH MODELZOO YOLOV3的训练和推理的实现。

YOLOV3网络结构GitHub链接可参考：[https://github.com/ultralytics/yolov3/tree/8bc9f56564f94bc59dab5a2f22935bbdbeb5774e](https://github.com/ultralytics/yolov3/tree/8bc9f56564f94bc59dab5a2f22935bbdbeb5774e)。


## 支持情况
### 模型训练支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  |
----- | ----- | ----- | ----- | ----- |
YOLOV3  | PyTorch  | MLU370-X8  | AMP/FP32  | Yes  |


### 模型推理支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | 
Supported Infer Mode
----- | ----- | ----- | ----- | ----- | 
YOLOV3  | PyTorch  | MLU370-S4/X4  | FP32  | cnnl |


## 默认参数配置
### 模型训练默认参数
### Optimizer
模型默认优化器为SGD，以下为相关参数：
* Momentum: 0.9
* Learning Rate: 0.2 for batch size 64
* Learning rate schedule: Linear schedule
* Weight decay: 1e-4
* Label Smoothing: None
* Epoch: 90

### 数据增强
模型使用了以下数据增强方法：
* 训练
    * Normolization
    * Crop image to 224*224
    * RandomHorizontalFlip
* 验证
    * Normolization
    * Crop image to 256*256
    * Center crop to 224*224

### 模型推理默认参数
* modeldir: 没有指定情况下，默认使用 torchvision 的预训练模型，可选指定训练完成的权重(eg. --modeldir /model/xxxx.pt)
* batch_size: 10,32,64 (batch_size <= 64 in MLU370s4)
* input_data_type：默认使用 float32
* jit&jit_fuse：默认开启
* qint：默认不开启量化
* save_result：默认开启推理结果保存于 result.json 文件


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

## 快速使用指南
### 件说明
* run_scripts/ 包含一键训练和推理的shell脚本文件
* models/ 包含原始模型仓库文件
* * AttModel.py tranformer网络结构定义
* * datasets.py 载入和打包训练数据
* * util.py 工具函数
* * train.py 模型训练脚本，更多信息使用python train.py -h查看
* * eval.py 模型推理验证脚本, 更多信息使用python eval.py -h查看

### 准备数据集
该YOLOV3脚本基于coco2014训练，数据集下载：<https://www.kaggle.com/datasets/nadaibrahim/coco2014> 。目录结构为：
```
├── annotations
│   ├── captions_train2014.json
│   ├── captions_val2014.json
│   ├── instances_train2014.json
│   ├── instances_val2014.json
│   ├── person_keypoints_train2014.json
│   └── person_keypoints_val2014.json
├── images
│   ├── test2014 -> ../test2014/
│   ├── train2014 -> ../train2014
│   └── val2014 -> ../val2014/
├── test2014
├── train2014
├── train2014.txt
├── train2014.zip
├── val2014
├── val2014.shapes
└── val2014.txt
```
指定数据集和模型权重路径：
```bash 
export PYTORCH_TRAIN_DATASET=/path/to/dataset
export PYTORCH_TRAIN_CHECKPOINT=/path/to/ckpt
```

### 准备模型
下载darknet53.conv.74到PYTORCH_TRAIN_CHECKPOINT/yolov3/darknet53.conv.74
```bash 
wget -c https://pjreddie.com/media/files/darknet53.conv.74 -O $PYTORCH_TRAIN_CHECKPOINT/yolov3/darknet53.conv.74
```

### 环境准备
#### 基于base docker image安装
##### 1、导入镜像
```
##下载Cambricon PyTorch docker镜像
docker load -i xxx.tar.gz
```

##### 2、启动测试容器（指定镜像名）（请自行指定数据集和权重挂载目录）
```
export IMAGE_NAME=YOUR_IMAGE_NAME
bash run_docker.sh [CONTAINER_NAME]
```

##### 3、配置容器环境

```
source env.sh
source /torch/venv3/pytorch/bin/activate
pip install -r models/requirements.txt
```

#### 使用Dockerfile准备环境
#### 1、构建 docker 镜像

```
export IMAGE_NAME=demo_yolov3
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../
```

####  2、创建并启动容器（请自行指定数据集和权重挂载目录）

```
docker run -it --ipc=host -v /data:/data -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name mlu_yolov3 --network=host $IMAGE_NAME
```

##### 3、配置容器环境

```
source env.sh
source /torch/venv3/pytorch/bin/activate
```

#### **一键执行训练脚本**
Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
Yolov3  | PyTorch  | MLU370-X8  | FP32  | 4  | bash run_scripts/YOLOV3_FP32_4MLU_Train.sh
Yolov3  | PyTorch  | MLU370-X8  | AMP  | 4  | bash run_scripts/YOLOV3_AMP_4MLU_Train.sh


#### ** 一键执行推理脚本**
Models  | Framework  | MLU   | Data Precision  |Run
----- | ----- | ----- | ----- | ----- | 
Yolov3  | PyTorch  | MLU370-S4  | FP32  | bash run_scripts/YOLOV3_Infer.sh

## **结果展示**

### ** 推理结果**
##### Infering accuracy results: MLU370-S4
Models | batch_size | (FP32) Mean AP   |
----- | ----- | ----- |
Yolov3  | 8 | 43.9 |

## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
* 数据集下载链接：[https://www.kaggle.com/datasets/nadaibrahim/coco2014](https://www.kaggle.com/datasets/nadaibrahim/coco2014)
* YOLOV3网络结构GitHub链接：[https://github.com/ultralytics/yolov3/tree/8bc9f56564f94bc59dab5a2f22935bbdbeb5774e](https://github.com/ultralytics/yolov3/tree/8bc9f56564f94bc59dab5a2f22935bbdbeb5774e)。
* 下载darknet53 weights (first 75 layers only)链接： https://pjreddie.com/media/files/darknet53.conv.74

## Release_Notes
@TODO
