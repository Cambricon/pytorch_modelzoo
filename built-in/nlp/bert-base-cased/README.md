# Bert-Base-Cased
---
## 模型概述
  基于[Hugging Face Transformers](https://github.com/huggingface/transformers)的bert-base-cased模型的MLU实现

## 支持情况
---
### 模型训练支持情况
| Models          | Framework | Supported MLU | Supported Data Precision | Multi-GPUs | 
| --------------- | --------- | ------------- | ------------------------ | ---------- | 
| Bert-Base-Cased | PyTorch   | MLU370-X8     | FP32/AMP                 | Yes        | 

### 模型推理支持情况
| Models          | Framework | Supported MLU | Supported Data Precision |
| --------------- | --------- | ------------- | ------------------------ |
| Bert-Base-Cased | PyTorch   | MLU370-S4/X4  | FP32                     |

## 默认参数配置
---
### 模型训练默认参数

#### Optimizer
模型优化器为transformer包提供的AdamW，参数配置如下：
- Learning Rate: 4e-5 for batch size 16
- epsilon=1e-8
- weight_decay: 0.0
- Epoch:2


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
- run_scripts/ 目录包含一键训练和推理的shell脚本文件
- models/ 目录包含原始模型仓库文件
  - `run_squad.py`:训练和推理脚本，更多信息使用`python run_squad.py -h`查看
### 准备数据集和预训练模型
下载数据集，[SQuADv1.1](https://rajpurkar.github.io/SQuAD-explorer/) 并解压，包含的内容如下：
```
|
|——dev-v1.1.json  
|——evaluate-v1.1.py  
|——train-v1.1.json  
|——train-v1.1.json_bert-base-cased_384_128_64  
|——train-v1.1.json_bert-large-uncased_384_128_64
```
指定数据集环境变量：
```bash
export SQUAD_DIR=YOUR_DATASET_PATH
```
### 环境准备
#### 基于base docker image安装
1. 导入镜像
```
##下载Cambricon PyTorch docker镜像
docker load -i xxx.tar.gz
```
2. 启动测试容器（根据下载的镜像名修改脚本）
```bash
export IMAGE_NAME=YOUR_IMAGE_NAME
bash run_docker.sh [CONTAINER_NAME]
```
3. 配置容器环境
```bash
source env.sh
source /torch/venv3/pytorch/bin/activate
pip install -r requirements.txt
```
#### 使用Dockerfile 准备环境
1. 构建 docker 镜像
```bash
export IMAGE_NAME=demo_bert_base_cased
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../
```
2. 创建并启动容器（指定数据集挂载目录）
```bash
docker run -it --ipc=host -v /data:/data -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name mlu_bert_base_cased --network=host $IMAGE_NAME
```
3. 配置容器环境
```bash
source env.sh
source /torch/venv3/pytorch/bin/activate
```
### 执行训练或推理脚本
```bash
bash run_scripts/bert_base_cased_FP32_2E_4MLUs_Train.sh
```
## 一键训练脚本
| Models          | Framework | MLU       | Data Precision | Cards | Description                      | Run                                                            |
| -----------     | --------- | --------- | -------------- | ----- | -------------------------------- | -----------------------------------------------------------    |
| Bert-Base-Cased | PyTorch   | MLU370-X8 | FP32           | 4     | from scratch training use 4 MLUs | bash run_scripts/bert_base_cased_FP32_2E_4MLUs_Train.sh |
| Bert-Base-Cased | PyTorch   | MLU370-X8 | AMP            | 4     | from scratch training use 4 MLUs | bash run_scripts/bert_base_cased_AMP_2E_4MLUs_Train.sh  |

## 一键推理脚本
| Models          | Framework | MLU       | Data Precision | Description                | Run                                          |
| --------------- | --------- | --------- | -------------- | -------------------------- | -------------------------------------------- |
| Bert-Base-Cased | Pytorch   | MLU370-S4 | FP32           | inference script           | bash run_scripts/bert_base_cased_Infer.sh |
## 结果展示
Training accuracy result:MLU370-X8
| Models          | Data Precision | F1 Score |
| --------------- | -------------- | -------- |
| Bert-Base-Cased | FP32           | 87.62    |
| Bert-Base-Cased | AMP            | 87.03    |

## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

数据集下载链接：[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)
预训练模型下载链接：[Bert-Base-Cased](https://huggingface.co/bert-base-cased/tree/main)


## Release_Notes
@TODO
