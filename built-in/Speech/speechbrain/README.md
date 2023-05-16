# SpeechBrain

## 模型概述
  SpeechBrain是PyTorch的官方开源语音工具包，其可以完成自动语音识别，说话人识别，验证等任务。
  本仓库为SpeechBrain的MLU实现,GPU实现可参考仓库：https://github.com/speechbrain/speechbrain

## 支持情况
---
### 模型训练支持情况
| Models      | Framework | Supported MLU | Supported Data Precision | Multi_GPUs |
| ----------- | --------- | ------------- | ------------------------ | ---------- |
| ECAPA-TDNN      | PyTorch1.9   |   MLU370-X8   |   FP32               | Yes        |

### 模型推理支持情况
| Models      | Framework | Supported MLU | Supported Data Precision |
| ----------- | --------- | ------------- | ------------------------ |
| ECAPA-TDNN      | PyTorch1.9   | MLU370-S4/X4  |    FP32              |


## 默认参数配置
以下为enet模型的默认参数配置：

Models  | 优化器  | Learning Rate   | Learning rate schedule |  Weight decay |
---- | ----- | ----- | ----- | ----- |
ECAPA-TDNN  | Adam  | 1e-3  | CyclicLR | 2e-5 |



## 环境依赖
* Linux常见操作系统版本(如Ubuntu18.04，Ubuntu20.04, CentOS7.6)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算板卡MLU370-X8;
* Cambricon Driver >=v4.20.6；
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 快速入门指南

### 文件说明
- run_scripts/ 包含一键训练和推理的shell脚本文件
- models/ 包含原始模型仓库文件
  - speechbrain： speechbrain实现源码，安装后可以直接使用
  - setup.py：    speechbrain安装脚本
  - recipes：     基于speechbrain，针对各种数据集的训推一体实现代码


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
#1、env.sh中的`PYTORCH_TRAIN_DATASET`为容器内训练数据集的根路径，这个环境变量需要用户根据真实情况设置。 

source env.sh
pip install -r models/requirements.txt
cd models
python setup.py install
cd -
```


#### 使用Dockerfile准备环境
#### 1、生成speechbrain的Docker镜像：

```
export IMAGE_NAME=test_speech_brain
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../
```

####  2、创建容器

```
#注意：前一个path_of_dataset为用户host主机端数据集存放的路径，后一个path_of_dataset为映射到镜像内的路径。

docker run -it --ipc=host -v path_of_dataset:path_of_dataset -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name test_speech_brain_container --network=host $IMAGE_NAME
```

##### 3、启动虚拟环境，设置环境变量

```
#1、env.sh中的`PYTORCH_TRAIN_DATASET`为容器内训练数据集的根路径，这个环境变量需要用户根据真实情况设置。 

source env.sh
```

### 数据集准备
现阶段本仓库只支持了SpeechBrain包中的ECAPA_TDNN模型，该模型是基于VoxCeleb数据集训练的，下载链接：<https://www.robots.ox.ac.uk/~vgg/data/voxceleb/>。数据集请放在`$PYTORCH_TRAIN_DATASET`目录下。目录结构为：
```
├── voxceleb1_all
│   ├── id10001
│   ├── id10002
│   ├── ...
├── voxceleb2
│   ├── id00012
│   ├── id00015
│   ├── ...
├── voxceleb_wav
│   ├── RIRS_NOISES
│   ├── meta
│   ├── ...
```

### Run 脚本执行
```
bash run_scripts/ECAPA_TDNN_FP32_8MLUs_Train.sh
```

#### 一键执行训练脚本

>Attention:用户需要实际情况修改models/recipes/VoxCeleb/SpeakerRec/hparams/train_ecapa_tdnn.yaml中的数据集路径`data_folder`以及checkpoint保存路径`output_folder`

Models  | Framework  | MLU   | MODE  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
ECAPA_TDNN  | PyTorch1.9  | MLU370-X8  | FP32(from scratch)  | 8 | bash run_scripts/ECAPA_TDNN_FP32_8MLUs_Train.sh |

#### 一键执行推理脚本

>Attention:用户需要实际情况修改models/recipes/VoxCeleb/SpeakerRec/hparams/verification_ecapa.yaml中的数据集路径`data_folder`以及推理checkpoint路径`output_folder`

Models  | Framework  | MLU   |Run
----- | ----- | ----- | ----- | 
ECAPA_TDNN  | PyTorch1.9  | MLU370-S4/MLU370-X4 | bash run_scripts/ECAPA_TDNN_Infer.sh

## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
