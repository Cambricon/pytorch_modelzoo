# WaveRNN(Pytorch)
---
## 模型概述
  [WaveRNN](https://arxiv.org/pdf/1802.08435.pdf) 是谷歌提出的语音合成算法，模型中使用了较为先进的计算方法去提高生成语音的质量同时保证模型很小，可应用在手机、嵌入式等资源比较少的系统。
  本仓库为 WaveRNN 的MLU实现，原始GPU实现仓库为: [WaveRNN](https://github.com/fatchord/WaveRNN)。

## 支持情况
---
### 训练模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-MLUs |
----- | ----- | ----- | ----- | ----- |
WaveRNN  | PyTorch1.6  | MLU370-X8  | AMP/FP32  | Yes  |

### 推理模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Supported Infer Mode | 
----- | ----- | ----- | ----- | ----- |
WaveRNN  | PyTorch1.6  | MLU370-X8  | FP32      | CNNL |

## 默认参数配置
---
### Optimizer
Models  | 优化器  | Learning Rate   | Learning rate schedule |  Weight decay | Epoch
---- | ----- | ----- | ----- | ----- | ----- |
WaveRNN  | Adam  | 1e-4  | None  | 0 | 10

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

## 快速入门指南
---
### 文件说明
- run_scripts/ 包含一键训练和推理的shell脚本文件
- models/ 包含原始模型仓库文件
- `train.py` 模型训练入口，更多信息使用`python train.py -h`查看
- `val.py` 模型推理入口，更多信息使用`python val.py -h`查看

### 准备数据集
下载数据集 [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)，并解压，解压命令可以使用：
 ```bash 
tar -xvf LJSpeech-1.1.tar.bz2 --no-same-owner
 ```
解压后的数据集放置在 `WaveRNN/models/dataset` 目录下，目录结构为：
```bash 
models/dataset/
└── LJSpeech-1.1
    ├── README
    ├── metadata.csv
    └── wavs
``` 

### 环境准备
#### 基于base docker image安装
##### 1、导入镜像
```bash
##下载Cambricon PyTorch docker镜像
docker load -i xxx.tar.gz
```

##### 2、启动测试容器
```bash
## run_docker.sh中的path_of_pytorch_modelzoo:path_of_pytorch_modelzoo中，
## 前一个path_of_pytorch_modelzoo为用户host主机端pytorch_modelzoo真实路径，
## 后一个path_of_pytorch_modelzoo为映射到容器内的路径。

## 默认的 IMAGE_NAME 已设置为 yellow.hub.cambricon.com/pytorch/pytorch:v1.8.0-torch1.6-ubuntu18.04-py37
## 默认的 MY_CONTAINER 已设置为 wavernn_pytorch_1_6_0

bash run_docker.sh
```

##### 3、在容器中安装依赖
```bash
pip install -r models/requirements.txt
```

#### 使用Dockerfile准备环境
##### 1、构建 docker 镜像
```bash
export IMAGE_NAME=test_wavernn_pytorch_1_6_0
## ../../../路径下包含tools/  built-in/ 等文件夹。
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../
```

##### 2、创建并启动容器

```bash
## 注意：默认的容器名 name 已设置为 test_mlu_wavernn

docker run -it --ipc=host -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name test_mlu_wavernn --network=host $IMAGE_NAME
```

### 执行训练或推理脚本
```bash
bash run_scripts/WaveRNN_FP32_10E_4MLUs_Train.sh
```

## 一键训练脚本
---
注意：训练过程中，生成的权重和log默认保存在 `models/output` 路径下，由 --output 指定。
| Models      | Framework | MLU       | Data Precision | Cards | Description                      | Run                                                         |
| ----------- | --------- | --------- | -------------- | ----- | -------------------------------- | ----------------------------------------------------------- |
| WaveRNN | PyTorch1.6| MLU370-X8 | FP32          | 4     | training use 4 MLUs  | run_scripts/WaveRNN_FP32_10E_4MLUs_Train.sh  |
| WaveRNN | PyTorch1.6| MLU370-X8 | AMP           | 4     | training use 4 MLUs  | run_scripts/WaveRNN_AMP_10E_4MLUs_Train.sh   |

## 一键推理脚本
---
注意: 默认使用的推理权重为 --checkpoint-path 指定的 `./output/checkpoint_WaveRNN_10.pt`，可在 `run_scripts/WaveRNN_Infer.sh` 中手动修改权重路径。 

| Models      | Framework | MLU       | Data Precision | Description                | Run                                          |
| ----------- | --------- | --------- | -------------- | -------------------------- | -------------------------------------------- |
| WaveRNN | PyTorch1.6  | MLU370-X8 | FP32           | inference script           | bash run_scripts/WaveRNN_Infer.sh |


## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

## Release_Notes
@TODO
