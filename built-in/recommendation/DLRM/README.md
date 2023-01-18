# DLRM
---
## 模型概述

  本仓库为DLRM的MLU实现，GPU实现可参考仓库: https://github.com/mlcommons/training/tree/8e7ad54541aeda54a8e5152732b9fb293a22b10c/recommendation

## 支持情况
---
### 模型训练支持情况
| Models      | Framework  | Supported MLU | Supported Data Precision | Multi_GPUs |
| ----------- | ---------- | ------------- | ------------------------ | ---------- |
| DLRM        | PyTorch1.6 | MLU370-X8     | FP32, AMP                | Yes        |

### 模型推理支持情况
| Models      | Framework  | Supported MLU | Supported Data Precision | Supported Infer Mode |
| ----------- | ---------- | ------------- | ------------------------ |----------------------|
| DLRM        | PyTorch1.6 | MLU370-S4/X4  | FP32                     | cnnl                 |

## 默认参数配置
---
### 模型训练默认参数

#### Optimizer
模型默认优化器为Adam，参数配置如下：
- Learning Rate: 0.0002 for batch size 65536
- layers = 256 256 128 64, seed = 0, factors = 64
- Epoch:20


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
## 快速使用指南
---
### 文件说明
- run_scripts/ 包含一键训练和推理的shell脚本文件
- models/ 包含原始模型仓库文件
  - `data_generation` 数据生成脚本，更多信息查看`data_generation/fractal_graph_expansions`中的README
  - `recommendation` 数据下载脚本与网络脚本
### 准备数据集
下载数据集[MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/)，原始数据集使用`data_generation`中的脚本扩展成 4x users 和 16x items。要获取扩展数据集，请按照`data_generation/fractal_graph_expansions` 目录中 README 文件中`Running instructions`部分进行操作。

指定数据集和模型权重路径：
```bash
export PYTORCH_TRAIN_DATASET=/path/to/dataset
```
### 环境准备
#### 基于base docker image安装
1. 导入镜像
```
##下载Cambricon PyTorch docker镜像
docker load -i xxx.tar.gz
```
2. 启动测试容器（指定镜像名）
```bash
export IMAGE_NAME=YOUR_IMAGE_NAME
bash run_docker.sh [CONTAINER_NAME]
```
3. 配置容器环境
```bash
source env.sh
source /torch/venv3/pytorch/bin/activate
pip install -r models/recommendation/pytorch/requirements.txt
```
#### 使用Dockerfile 准备环境
1. 构建 docker 镜像
```bash
export IMAGE_NAME=demo_dlrm
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../
```
2. 创建并启动容器（请自行指定数据集和权重挂载目录）
```bash
docker run -it --ipc=host -v /data:/data -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name mlu_dlrm --network=host $IMAGE_NAME
```
3. 配置容器环境
```bash
source env.sh
source /torch/venv3/pytorch/bin/activate
```
### 执行训练或推理脚本
```bash
bash run_scripts/DLRM_AMP_20E_4MLUs_Train.sh
```
## 一键训练脚本
| Models      | Framework | MLU       | Data Precision | Cards | Description                      | Run                                                         |
| ----------- | --------- | --------- | -------------- | ----- | -------------------------------- | ----------------------------------------------------------- |
| DLRM        | PyTorch1.6| MLU370-X8 | FP32           | 4     | from scratch training use 4 MLU  | bash run_scripts/DLRM_FP32_20E_4MLUs_Train.sh               |
| DLRM        | PyTorch1.6| MLU370-X8 | AMP            | 4     | from scratch training use 4 MLU  | bash run_scripts/DLRM_AMP_20E_4MLUs_Train.sh                |

## 一键推理脚本
> Attention: 执行推理脚本前需自行训练得到模型权重,默认权重保存在`models/recommendation/pytorch/ckp`目录下，并修改推理脚本中的权重路径即环境变量`PYTORCH_INFER_CHECKPOINT`

| Models      | Framework | MLU       | Data Precision | Description                | Run                                          |
| ----------- | --------- | --------- | -------------- | -------------------------- | -------------------------------------------- |
| DLRM        | PyTorch1.6| MLU370-S4 | FP32           | inference script           | bash run_scripts/DLRM_infer.sh               |

## 结果展示
Training accuracy result:MLU370-X4
| Models      | Epochs | FP32 HR@10      |
| ----------- | ------ | --------------- |
| DLRM        | 20     | 0.5568          |

## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

## Release_Notes
@TODO
