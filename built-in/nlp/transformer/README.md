# Transformer
---
## 模型概述
  Transformer网络源于论文[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

  本仓库为Transformer的MLU实现，GPU实现可参考仓库: https://github.com/leviswind/pytorch-transformer

## 支持情况
---
### 模型训练支持情况
| Models      | Framework  | Supported MLU | Supported Data Precision | Multi_GPUs |
| ----------- | ---------- | ------------- | ------------------------ | ---------- |
| Transformer | PyTorch1.6 | MLU370-X8     | FP32                     | Yes        |

### 模型推理支持情况
| Models      | Framework  | Supported MLU | Supported Data Precision | Supported Infer Mode |
| ----------- | ---------- | ------------- | ------------------------ |----------------------|
| Transformer | PyTorch1.6 | MLU370-S4/X4  | FP32                     | cnnl                 |

## 默认参数配置
---
### 模型训练默认参数

#### Optimizer
模型默认优化器为Adam，参数配置如下：
- Learning Rate: 0.0001 for batch size 32
- beta_1=0.9, beta_2=0.98, epsilon=1e-8
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
  - `AttModel.py` tranformer网络结构定义
  - `hyperparams.py` 网络所需用到的超参
  - `prepro.py` 用于生成 source 和 target 语料库文件
  - `data_load.py` 载入和打包训练数据
  - `modules.py` 网络需要用到的模块定义
  - `util.py` 工具函数
  - `train.py` 模型训练脚本，更多信息使用`python train.py -h`查看
  - `eval.py` 模型推理验证脚本, 更多信息使用`python eval.py -h`查看
### 准备数据集
下载数据集[IWSLT 2016 German–English parallel corpus](https://wit3.fbk.eu/2016-01)，并解压，包含的内容如下：
```
├── de-en
├── de-en.tgz
├── IWSLT16.TED.dev2010.de-en.de.xml
├── IWSLT16.TED.dev2010.de-en.en.xml
├── IWSLT16.TED.tst2010.de-en.de.xml
├── IWSLT16.TED.tst2010.de-en.en.xml
├── IWSLT16.TED.tst2011.de-en.de.xml
├── IWSLT16.TED.tst2011.de-en.en.xml
├── IWSLT16.TED.tst2012.de-en.de.xml
├── IWSLT16.TED.tst2012.de-en.en.xml
├── IWSLT16.TED.tst2013.de-en.de.xml
├── IWSLT16.TED.tst2013.de-en.en.xml
├── IWSLT16.TED.tst2014.de-en.de.xml
├── IWSLT16.TED.tst2014.de-en.en.xml
├── IWSLT16.TEDX.dev2012.de-en.de.xml
├── IWSLT16.TEDX.dev2012.de-en.en.xml
├── IWSLT16.TEDX.tst2013.de-en.de.xml
├── IWSLT16.TEDX.tst2013.de-en.en.xml
├── IWSLT16.TEDX.tst2014.de-en.de.xml
├── IWSLT16.TEDX.tst2014.de-en.en.xml
├── README
├── train.en
├── train.tags.de-en.de
└── train.tags.de-en.en
```
指定数据集和模型权重路径：
```bash
export IWSLT_CORPUS_PATH=/path/to/dataset
export TRANSFORMER_CKPT=/path/to/ckpt
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
pip install -r requirements.txt
```
#### 使用Dockerfile 准备环境
1. 构建 docker 镜像
```bash
export IMAGE_NAME=demo_transformer
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../
```
2. 创建并启动容器（请自行指定数据集和权重挂载目录）
```bash
docker run -it --ipc=host -v /data:/data -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name mlu_transformer --network=host $IMAGE_NAME
```
3. 配置容器环境
```bash
source env.sh
source /torch/venv3/pytorch/bin/activate
```
### 执行训练或推理脚本
```bash
bash run_scripts/Transformer_FP32_20E_8MLUs_Train.sh
```
## 一键训练脚本
| Models      | Framework | MLU       | Data Precision | Cards | Description                      | Run                                                         |
| ----------- | --------- | --------- | -------------- | ----- | -------------------------------- | ----------------------------------------------------------- |
| Transformer | PyTorch1.6| MLU370-X8 | FP32           | 1     | from scratch training use 1 MLU  | bash run_scripts/Transformer_FP32_20E_1MLU_Train.sh  |
| Transformer | PyTorch1.6| MLU370-X8 | FP32           | 8     | from scratch training use 8 MLUs | bash run_scripts/Transformer_FP32_20E_8MLUs_Train.sh |

## 一键推理脚本
> Attention: 执行推理脚本前需自行训练得到模型权重，并修改推理脚本中的权重路径

| Models      | Framework | MLU       | Data Precision | Description                | Run                                          |
| ----------- | --------- | --------- | -------------- | -------------------------- | -------------------------------------------- |
| Transformer | PyTorch1.6  | MLU370-S4 | FP32           | inference script           | bash run_scripts/Transformer_Infer.sh |

## 结果展示
Training accuracy result:MLU370-X8
| Models      | Epochs | FP32 Bleu Score |
| ----------- | ------ | --------------- |
| Transformer | 10     | 16.23           |
| Transformer | 20     | 15.11           |

## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

数据集下载链接：[IWSLT 2016 German–English parallel corpus](https://wit3.fbk.eu/download.php?release=2016-01&type=texts&slang=de&tlang=en)

## Release_Notes
@TODO
