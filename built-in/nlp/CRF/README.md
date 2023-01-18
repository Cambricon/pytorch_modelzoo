# CRF
---
## 模型概述
CRF是Conditional random field,即条件随机场,CRF是一个序列化标注算法（sequence labeling algorithm），接收一个输入序列 并且输出目标序列，也能被看作是一种seq2seq模型。
例如，在词性标注任务中，输入序列为一串单词，输出序列就是相应的词性。除了词性标注之外，CRF还可以用来做chunking，命名实体识别等任务。


本仓库为CRF的MLU实现，GPU实现可参考仓库: https://github.com/kmkurn/pytorch-crf

## 支持情况
---
### 模型训练支持情况
| Models      | Framework | Supported MLU | Supported Data Precision | Multi_MLUs |
| ----------- | --------- | ------------- | ------------------------ | ---------- |
| CRF         | PyTorch   | MLU370-X8     | FP32                     | No         |


## 默认参数配置
---
### 模型训练默认参数

#### Optimizer
---

## 环境依赖
---
- Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
- 服务器装配好寒武纪计算版本MLU370-X8;
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
`models/` ：原始模型仓库代码
  - `main.py`: Serves as the entry point to the application. Encapsulates the training and testing routine.
  - `download_dataset.py`: download dataset from remote.
  - `torchcrf/` folder contains information about the building blocks of CRF and the way they are assembled. Its contents are:
  - `__init__.py`: Defines the different blocks that are used to assemble CRF.

### 准备数据集
训练与测试使用Penn Treebank，由于数据集收费，本仓库使用nltk包中的约5%的开源Penn Treebank代替。数据集目录结构如下：
/xxx/xxx/xxx/treebank/
└── corpora
      ├── treebank
      │   ├── combined
### 方法一:
未下载数据集可在容器环境准备完后再运行如下命令下载
```bash
mkdir -p data/nltk_data
python models/download_dataset.py --data ./data/nltk_data
```
### 方法二:

```
在创建docker容器前已下载好数据集，可选择将数据集目录挂载到容器中，并设置环境变量：

```bash
export CRF_DATASET=YOUR DATA PATH
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
source /torch/venv3/pytorch/bin/activate
pip install -r requirements.txt
```
#### 使用Dockerfile 准备环境
1. 构建 docker 镜像
```bash
export IMAGE_NAME=demo_crf
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../
```
2. 创建并启动容器（指定数据集挂载目录）
```bash
docker run -it --ipc=host -v /data:/data -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name mlu_crf --network=host $IMAGE_NAME
```
3. 配置容器环境
```bash
source /torch/venv3/pytorch/bin/activate
source env.sh
```

### 执行脚本
CRF不是一个完整的网络,所有也没有反向操作,只是跑了前向。通过在mlu上测试,发现跑50次的时候相对来说比较稳定,当然次数越多肯定也会越稳定。
```bash
bash test/test_benchmark.sh fp32-mlu
```

## 结果展示

在MLU370-X8下将CRF跑50次前向,然后计算出per-word-error的mean和std.
| 实验次数 | iters |  avg of per-word-error rate  |  std of per-word-error rate  |
| :------:| :--------------------: | -------- | -------- |
| 1 | 50 | 0.0516  | 0.000858 |
| 2 | 50 | 0.0517  | 0.001019 |
| 3 | 50 | 0.0517  | 0.000971 |
| 4 | 50 | 0.0518  | 0.000987 |
| 5 | 50 | 0.0517  | 0.000967 |


## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。


## Release_Notes
@TODO
