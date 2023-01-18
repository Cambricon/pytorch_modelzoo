# DeepSpeech2
---
## 模型概述
  DeepSpeech2网络源于论文[Deep Speech 2](https://arxiv.org/pdf/1512.02595.pdf)

  本仓库为DeepSpeech2的MLU实现，GPU实现可参考仓库: https://github.com/mlcommons/training/tree/0badcd1786fcb007725ed05f1c44e9d80bbeac52/speech_recognition

## 支持情况
---
### 模型训练支持情况
| Models      | Framework  | Supported MLU | Supported Data Precision | Multi_GPUs |
| ----------- | ---------- | ------------- | ------------------------ | ---------- |
| DeepSpeech2 | PyTorch1.6 | MLU370-X8     | FP32                     | Yes        |

## 默认参数配置
---
### 模型训练默认参数
见./models/pytorch/params.py
#### Optimizer
模型默认优化器为SGD，参数配置如下：
- Learning Rate: 0.0001 for batch size 8
- momentum=0.9, weight_decay=0, nesterov=True
- Epoch:10

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
  - `./data/` 载入和打包训练数据以及数据集下载脚本
  - `./pytorch/model.py` deepspeech2网络结构定义
  - `./pytorch/params.py` 网络所需用到的超参
  - `./pytorch/decoder.py` 计算wer、cer
  - `./pytorch/eval_model.py` 训练中接入的推理模型
  - `./pytorch/train.py` 模型训练脚本，更多信息使用`python train.py -h`查看
### 准备数据集
详情见models/README_origin.md第4节。
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
pip install -r ./models/requirements.txt
```
另外，还需要具有sudo权限安装libsndfile模块，自动安装直接执行./models/lib.sh脚本，手动安装见说明文件./models/README.md。

#### 使用Dockerfile 准备环境
1. 构建 docker 镜像
```bash
export IMAGE_NAME=demo_deepspeech2
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../
```
2. 创建并启动容器（请自行指定数据集和权重挂载目录）
```bash
docker run -it --ipc=host -v /data:/data -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name mlu_deepspeech2 --network=host $IMAGE_NAME
```
3. 配置容器环境
```bash
source env.sh
source /torch/venv3/pytorch/bin/activate
```
### 执行训练或推理脚本
```bash
bash run_scripts/DeepSpeech2_FP32_10E_16MLU_Train.sh
```
其中``--save_folder``表示训练过程中保存的ckpt文件路径；``--model_path``表示训练完成后保存的ckpt文件路径。

## 一键训练脚本
| Models      | Framework | MLU       | Data Precision | Cards | Description                      | Run                                                         |
| ----------- | --------- | --------- | -------------- | ----- | -------------------------------- | ----------------------------------------------------------- |
| DeepSpeech2 | PyTorch1.6| MLU370-X8 | FP32           | 16     | from scratch training use 16 MLU  | bash run_scripts/DeepSpeech2_FP32_10E_16MLU_Train.sh  |

## 一键推理脚本
NA

## 结果展示
Training accuracy result:MLU370-X8
| Models      | Epochs | FP32 avg loss |
| ----------- | ------ | ------------- |
| DeepSpeech2 | 10     | 0.3443        |

## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

## Release_Notes
@TODO
