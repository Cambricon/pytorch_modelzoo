# BERT
---
## 模型概述
  本仓库是基于[NVIDIA BERT For PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)的MLU实现

## 支持情况
---
### 模型训练支持情况
| Models | Framework  | Supported MLU | Supported Data Precision | Multi-GPUs |
| ------ | ---------- | ------------- | ------------------------ | ---------- |
| BERT   | PyTorch1.6 | MLU370-X8     | FP32                     | Yes        |

### 模型推理支持情况
| Models | Framework  | Supported MLU | Supported Data Precision |
| ------ | ---------- | ------------- | ------------------------ |
| BERT   | PyTorch1.6 | MLU370-S4/X4  | FP32                     |

## 默认参数配置
---
### 模型训练默认参数

#### Optimizer
模型优化器为BERT定制的Adam，参数配置如下：
- Learning Rate: 3e-5 for batch size 4
- epsilon=1e-6
- weight_decay: 0.01
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
  - `run_squad.py`:训练和推理文件，更多信息使用`python run_squad.py -h`查看
  - `run_squad.sh`:训练和推理脚本，对`run_squad.py`的封装
### 准备数据集
下载数据集[SQuADv1.1](https://rajpurkar.github.io/SQuAD-explorer/) 并解压（也可以通过仓库models/data/squad/squad_download.sh脚本下载），包含的内容如下：
```
|
|——dev-v1.1.json  
|——evaluate-v1.1.py  
|——train-v1.1.json  
```
指定数据集环境变量：
```bash
export SQUAD_DIR=YOUR_DATASET_PATH
```
### 基于base docker image安装

1. 导入镜像
```
##下载Cambricon PyTorch docker镜像
docker load -i xxx.tar.gz
```
2. 启动测试容器（根据下载的镜像名修改脚本）
```bash
#修改run_docker.sh中的/your/data:/your/data，其中
#前一个/your/data为用户host主机端data真实路径，
#后一个/your/data为映射到容器内的路径。
bash run_docker.sh
```
3. 配置容器环境
```bash
#1、env.sh中的`SQUAD_DIR`为容器内squad训练数据集的路径，这个环境变量需要用户根据真实情况设置。 
#2、env.sh中的`BERT_MODEL`为BERT网络预训练权重，这个环境变量需要用户根据真实情况设置，下载链接和转换步骤见下面准备预训练模型部分。 
#3、env.sh中的`BERT_INFER_MODEL`为BERT网络推理时的权重，这个环境变量需要用户根据真实情况设置。
source env.sh
source /torch/venv3/pytorch/bin/activate
```
#### 使用Dockerfile 准备环境
1. 构建 docker 镜像
```bash
export IMAGE_NAME=demo_bert
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../
```
2. 创建并启动容器（指定数据集挂载目录）
```bash
docker run -it --ipc=host -v /data:/data -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name mlu_bert --network=host $IMAGE_NAME
```
3. 配置容器环境
```bash
#1、env.sh中的`SQUAD_DIR`为容器内squad训练数据集的路径，这个环境变量需要用户根据真实情况设置。 
#2、env.sh中的`BERT_MODEL`为BERT网络预训练权重，这个环境变量需要用户根据真实情况设置，下载链接和转换步骤见下面准备预训练模型部分。 
#3、env.sh中的`BERT_INFER_MODEL`为BERT网络推理时的权重，这个环境变量需要用户根据真实情况设置。
source env.sh
source /torch/venv3/pytorch/bin/activate
```

### 准备预训练模型

下载预训练模型[BERT](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)并解压，包含内容如下：

```
.
├── bert_config.json
├── bert_model.ckpt.data-00000-of-00001
├── bert_model.ckpt.index
├── bert_model.ckpt.meta
└── vocab.txt
```

下载的预训练模型是TF的checkpoint，需要通过models/convert_bert_original_tf_checkpoint_to_pytorch.py 脚本转换为PyTorch的checkpoint，转换前需要先安装以下依赖

```
pip install protobuf==3.20.0
pip install tensorflow==1.15.0
```

调用脚本进行转换

```
python convert_bert_original_tf_checkpoint_to_pytorch.py --tf_checkpoint_path /Path/to/bert_model.ckpt --bert_config_file /Path/to/bert_config.json --pytorch_dump_path /Path/to/pytorch_model.pt
```

指定预训练模型和推理模型环境变量：

```
export BERT_MODEL=/Path/to/pytorch_model.pt
export BERT_INFER_MODEL=/Path/to/infer_pytorch_model.bin #一般为Training的output路径下
```

### 执行训练或推理脚本

```bash
bash run_scripts/BERT_FP32_2E_4MLUs_Train.sh
```
## 一键训练脚本
| Models | Framework | MLU       | Data Precision | Cards | Description                  | Run                                          |
| ------ | --------- | --------- | -------------- | ----- | ---------------------------- | -------------------------------------------- |
| BERT   | PyTorch   | MLU370-X8 | FP32           | 4     | finetune training use 4 MLUs | bash run_scripts/BERT_FP32_2E_4MLUs_Train.sh |

## 一键推理脚本

在推理前需要设置BERT_INFER_MODEL环境变量，指定需要推理的模型，如果是经过上面训练脚步执行后的，模型位置一般在models/output/pytorch_model.bin

| Models | Framework | MLU          | Data Precision | Description      | Run                            |
| ------ | --------- | ------------ | -------------- | ---------------- | ------------------------------ |
| BERT   | PyTorch   | MLU370-S4/X4 | FP32           | inference script | bash run_scripts/BERT_Infer.sh |
## 结果展示
Training accuracy result:MLU370-X8
| Models | Data Precision | F1 Score |
| ------ | -------------- | -------- |
| BERT   | FP32           | 88.69    |

## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

数据集下载链接：[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)
预训练模型下载链接：[BERT](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)


## Release_Notes
@TODO
