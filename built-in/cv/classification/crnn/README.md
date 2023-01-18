# Convolutional Recurrent Neural Network

## 模型概述
CRNN网络源于论文[An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)

本仓库为CRNN的MLU实现，GPU实现可参考仓库: [https://github.com/bgshih/crnn](https://github.com/bgshih/crnn)

## 支持情况
### 模型训练支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  |
----- | ----- | ----- | ----- | ----- |
CRNN  | PyTorch1.6  | MLU370-X8  | FP32  | Yes  |

### 推理模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision   | Supported Infer Mode | 
----- | ----- | ----- | ----- | ----- |
CRNN  | PyTorch1.6  | MLU370-S4/X4  | FP32  | cnnl | 

## 默认参数配置
### 模型训练默认参数配置

#### Optimizer
模型默认优化器为Adam，参数配置如下：
- Learning Rate: 0.0001 for batch size 32
- beta_1=0.9, beta_2=0.999, epsilon=1e-8
- Epoch:30

## 环境依赖
- Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
- 服务器装配好寒武纪计算板卡MLU370-X8;
- Cambricon Driver >=v4.20.6；
- CNToolKit >=2.8.3;
- CNNL >=1.10.2;
- CNCL >=1.1.1;
- CNLight >=0.12.0;
- CNPyTorch >= 1.3.0;
- 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 快速使用指南

### 文件说明
- run_scripts/ 包含一键训练和推理的shell脚本文件
- models/ 包含原始模型仓库文件
  - crnn.py crnn网络结构定义
  - dataset.py 读取数据集的函数
  - util.py 工具函数
  - train.py 模型训练脚本，更多信息使用python train.py -h查看
  - test.py 模型推理验证脚本, 更多信息使用python eval.py -h查看
  - warp-ctc mlu warp-ctc-loss工具

### 数据集准备
下载数据集：[https://www.kaggle.com/datasets/garvitchaudhary/mjsynth](https://www.kaggle.com/datasets/garvitchaudhary/mjsynth), 并解压到` $PYTORCH_TRAIN_DATASET/Synth90k`目录下。
指定数据集和模型权重路径：
```bash 
export PYTORCH_TRAIN_DATASET=path_of_dataset
export PYTORCH_TRAIN_CHECKPOINT=path_of_ckpt
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
source /torch/venv3/pytorch/bin/activate
pip install -r models/requirements.txt
```

#### 使用Dockerfile准备环境
#### 1、构建 docker 镜像

```
export IMAGE_NAME=demo_crnn
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../../
```

####  2、创建并启动容器（请自行指定数据集和权重挂载目录）

```
docker run -it --ipc=host -v /your/data:/your/data -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name mlu_crnn --network=host $IMAGE_NAME
```

##### 3、配置容器环境

```
source env.sh
source /torch/venv3/pytorch/bin/activate
```

### Run 脚本执行
```
bash run_scripts/CRNN_FP32_30E_4MLUs_Train.sh
```

#### 一键执行训练脚本
Models  | Framework  | MLU   | 
Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
CRNN  | PyTorch1.6  | MLU370-X8  | FP32  | 4  | bash run_scripts/CRNN_FP32_30E_4MLUs_Train.sh

#### 一键执行推理脚本
Models  | Framework  | MLU                 | Data Precision | Description      | Run                            |
-----   | ---------- | ------------------- | -------------- | ---------------- | -----------------------------  |
CRNN    | PyTorch1.6 | MLU370-S4/MLU370-X4 | FP32           | inference script | bash run_scripts/CRNN_Infer.sh |

## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

Synth90k 数据集下载链接：[https://www.kaggle.com/datasets/garvitchaudhary/mjsynth](https://www.kaggle.com/datasets/garvitchaudhary/mjsynth)

CRNN 模型代码链接：[https://github.com/bgshih/crnn](https://github.com/bgshih/crnn)


## Release_Notes
@TODO
