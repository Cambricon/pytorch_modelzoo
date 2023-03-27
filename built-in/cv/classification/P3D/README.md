# P3D(Pytorch)
---
## 模型概述
  P3D网络源于论文[Learning Spatio-Temporal Representation With Pseudo-3D Residual Networks](https://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf)

  本仓库为P3D的MLU实现，GPU实现可参考仓库: [pseudo-3d-pytorch](https://github.com/naviocean/pseudo-3d-pytorch/tree/50297d11248630792709782f467982e80c281384)

## 支持情况
### 训练模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-MLUS |
----- | ----- | ----- | ----- | ----- |
P3D  | PyTorch1.6  | MLU370-X8  | AMP/FP32  | Yes  |

### 推理模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision   | Supported Infer Mode | 
----- | ----- | ----- | ----- | ----- |
P3D  | PyTorch1.6  | MLU370-S4/X4  | FP16/FP32  | CNNL |

## 默认参数配置
以下为P3D模型的默认参数配置：

### Optimizer
Models  | 优化器  | Learning Rate   | Learning rate schedule |  Weight decay | Label Smoothing | Epoch
---- | ----- | ----- | ----- | ----- | ----- |---- |
P3D  | SGD  | 1e-3  | CyclicLR schedule  | 1e-4 | None | 60

### Data Augmentation
模型使用了以下数据增强方法：
* 训练
    * RandomSizedCrop,Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    * RandomHorizontalFlip,Randomly horizontally flips the given PIL.Image with a probability of 0.5
    * Normolization
* 验证
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
- `models/train.py` 模型训练脚本

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
#1、env.sh中的`PYTORCH_TRAIN_DATASET`为容器内ucf101训练数据集的路径，这个环境变量需要用户根据真实情况设置。 
#2、env.sh中的`PYTORCH_TRAIN_CHECKPOINT`为P3D网络预训练权重，这个环境变量需要用户根据真实情况设置，下载链接见末尾。 
#3、env.sh中的`PYTORCH_INFER_CHECKPOINT`为P3D网络推理时的权重，这个环境变量需要用户根据真实情况设置。 

source /torch/venv3/pytorch/bin/activate
source env.sh
```


#### 使用Dockerfile准备环境
#### 1、生成vision_classification的Docker镜像：

```
export IMAGE_NAME=test_p3d
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../
```

####  2、创建容器

```
#注意：前一个path_of_dataset为用户host主机端数据集存放的路径，后一个path_of_dataset为映射到镜像内的路径。

docker run -it --ipc=host -v path_of_dataset:path_of_dataset -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name test_p3d_classify --network=host $IMAGE_NAME
```

##### 3、启动虚拟环境，设置环境变量

```
#1、env.sh中的`PYTORCH_TRAIN_DATASET`为容器内ucf101训练数据集的路径，这个环境变量需要用户根据真实情况设置。 
#2、env.sh中的`PYTORCH_TRAIN_CHECKPOINT`为P3D网络预训练权重，这个环境变量需要用户根据真实情况设置，下载链接见末尾。
#3、env.sh中的`PYTORCH_INFER_CHECKPOINT`为P3D网络推理时的权重，这个环境变量需要用户根据真实情况设置。 

source /torch/venv3/pytorch/bin/activate
source env.sh
```

### 数据集准备
该P3D系列模型基于UCF101训练。
```
cd models/data
wget http://crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate
#建议创建一个文件夹保存UCF101.rar解压出来的avi视频文件
mkdir avi
#解压视频文件到avi文件夹下
unrar e UCF101.rar avi/
#开始制作数据集
python movefile.py
python makeVideoFolder.py
python extract.py

#根据训练代码需要，将train,test,validation，ucfTrainTestlist等训练需要的文件都放置于ucf101文件夹下
mkdir ucf101
mv train/ ucf101/
mv test/ ucf101/
mv validation/ ucf101/
mv ucfTrainTestlist/ ucf101/
```

数据集请放在` $PYTORCH_TRAIN_DATASET`目录下。目录结构为：
```
ucf101/
├── test
│   ├── ApplyEyeMakeup
    ├── ApplyLipstick
│   └── ...
├── train
│   ├── ApplyEyeMakeup
    ├── ApplyLipstick
│   └── ...
├── ucfTrainTestlist
│   ├── classInd.txt
    ├── testlist01.txt
│   └── ...
└── validation
    ├── ApplyEyeMakeup
    ├── ApplyLipstick
    └── ...
```

### Run 脚本执行
```
bash run_scripts/P3D_AMP_60E_8MLUs_Train.sh
```

#### 一键执行训练脚本
Models  | Framework  | MLU   | MODE  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
P3D  | PyTorch  | MLU370-X8  |  AMP(from scratch)  | 8  | bash run_scripts/P3D_AMP_60E_8MLUs_Train.sh
P3D  | PyTorch  | MLU370-X8  | FP32(from scratch)  | 8  | bash run_scripts/P3D_FP32_60E_8MLUs_Train.sh


#### 一键执行推理脚本
Models  | Framework  | MLU   |Run
----- | ----- | ----- | ----- | 
P3D  | PyTorch  | MLU370-S4/MLU370-X4 | bash run_scripts/P3D_Infer.sh


## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

ucf101 数据集下载链接：http://crcv.ucf.edu/data/UCF101/UCF101.rar      \
P3D 预训练模型下载链接：  \
1, P3D-199 trained on Kinetics dataset:     \
 [Google Drive url](https://drive.google.com/drive/folders/1u_l-yvhS0shpW6e0tCiqPE7Bd1qQZKdD) \
2, P3D-199 trianed on Kinetics Optical Flow (TVL1):       \
 [Google Drive url](https://drive.google.com/drive/folders/1u_l-yvhS0shpW6e0tCiqPE7Bd1qQZKdD)


## Release_Notes
@TODO

