# PointPillar(PyTorch)
## 模型概述

该sample包含PYTORCH MODELZOO PointPillar的训练和推理的实现。

PointPillar网络结构GitHub链接可参考：[`[PointRCNN]`](https://arxiv.org/abs/1812.04244), [`[Part-A2-Net]`](https://arxiv.org/abs/1907.03670), [`[PV-RCNN]`](https://arxiv.org/abs/1912.13192), [`[Voxel R-CNN]`](https://arxiv.org/abs/2012.15712) and [`[PV-RCNN++]`](https://arxiv.org/abs/2102.00463).


## 支持情况
### 模型训练支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  |
----- | ----- | ----- | ----- | ----- |
PointPillar  | PyTorch  | MLU370-X8  | FP32  | Yes  |


### 模型推理支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | 
Supported Infer Mode
----- | ----- | ----- | ----- | ----- | 
PointPillar  | PyTorch1.9  | MLU370-S4/X4  | FP32  | cnnl |


## 默认参数配置
### 模型训练默认参数
### Optimizer
模型默认优化器为adam_onecycle，以下为相关参数：
* Momentum: 0.9
* Learning Rate: 0.003 
* Weight decay: 0.01
* Epoch: 20

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
* * pcdet 包括数据集加载文件和模型文件等
* * tools 包括train.py test.py demo.py scipts等文件
* * data 数据集的txt文件
* * requirements.txt安装依赖
* * setup.py编译pcdet包

### 准备数据集
下载数据集<https://www.nuscenes.org/download> ，并解压。包含的内容如下：
```
├── maps
├── nuscenes_dbinfos_10sweeps_withvelo.pkl
├── nuscenes_infos_10sweeps_train.pkl
├── nuscenes_infos_10sweeps_val.pkl
├── nuscenes_infos_temporal_train.pkl
├── nuscenes_infos_temporal_val.pkl
├── samples
├── sweeps
├── v1.0-trainval
└── v1.0-trainval_meta.tgz
```
指定数据集和模型权重路径：
```bash 
export PYTORCH_TRAIN_DATASET=/path/to/dataset
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
export IMAGE_NAME=demo_pointpillar
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../
```

####  2、创建并启动容器（请自行指定数据集和权重挂载目录）

```
docker run -it --ipc=host -v /data:/data -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name mlu_pointpillar --network=host $IMAGE_NAME
```

##### 3、配置容器环境

```
source env.sh
source /torch/venv3/pytorch/bin/activate
```

#### **一键执行训练脚本**
Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
PointPillar  | PyTorch1.9  | MLU370-X8  | FP32  | 8  | bash run_scripts/PointPillar_FP32_8MLU_Train.sh


#### ** 一键执行推理脚本**
Models  | Framework  | MLU   | Data Precision  |Run
----- | ----- | ----- | ----- | ----- | 
PointPillar  | PyTorch1.9  | MLU370-S4  | FP32  | bash run_scripts/PointPillar_Infer.sh

## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
* 数据集下载链接：[https://www.kaggle.com/datasets/nadaibrahim/coco2014](https://www.kaggle.com/datasets/nadaibrahim/coco2014)
* PointPillar网络结构GitHub链接：[https://github.com/ultralytics/yolov3/tree/8bc9f56564f94bc59dab5a2f22935bbdbeb5774e](https://github.com/ultralytics/yolov3/tree/8bc9f56564f94bc59dab5a2f22935bbdbeb5774e)。
* 下载darknet53 weights (first 75 layers only)链接： https://pjreddie.com/media/files/darknet53.conv.74

## Release_Notes
@TODO
