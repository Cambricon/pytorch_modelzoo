# unet3D
---
## 模型概述
  unet3D网络源于论文[U-Net3D](https://arxiv.org/pdf/1606.06650.pdf)

  本仓库unet3D的MLU实现,GPU实现可参考仓库：https://github.com/mmarcinkiewicz/training/tree/master/image_segmentation/pytorch

## 支持情况
---
### 模型训练支持情况
| Models      | Framework | Supported MLU | Supported Data Precision | Multi_GPUs |
| ----------- | --------- | ------------- | ------------------------ | ---------- |
| unet3d      | PyTorch   | MLU370-X8     | FP32                     | Yes        |

### 模型推理支持情况
| Models      | Framework | Supported MLU | Supported Data Precision |
| ----------- | --------- | ------------- | ------------------------ |
| unet3d      | PyTorch   | MLU370-S4/X4  | FP32                     |


### 模型训练默认参数

#### Optimizer
模型默认优化器为sgd，参数配置如下：
- Epochs:1000


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
`run_scripts/`: 包含训练和推理的一键脚本
`models/`: 包含原始模型仓库的文件
  - `main.py`: Serves as the entry point to the application. Encapsulates the training routine.
  - `preprocess_data.py`: Converts the dataset to numpy format for training.
  - `evaluation_cases.txt`: A list of cases used for evaluation - a fixed split of the whole dataset.
  - `checksum.json`: A list of cases and their checksum for dataset completeness verification.
  
  - `data_loading/` folder contains the necessary load data. Its main components are:
    - `data_loader.py`: Implements the data loading.
    - `pytorch_loader.py`: Implements the data augmentation and iterators.
  
  - `model/` folder contains information about the building blocks of U-Net3D and the way they are assembled. Its contents are:
    - `layers.py`: Defines the different blocks that are used to assemble U-Net3D.
    - `losses.py`: Defines the different losses used during training and evaluation.
    - `unet3d.py`: Defines the model architecture using the blocks from the `layers.py` file.
  
  - `runtime/` folder contains scripts with training and inference logic. Its contents are:
    - `arguments.py`: Implements the command-line arguments parsing.
    - `callbacks.py`: Collection of performance, evaluation, and checkpoint callbacks.
    - `distributed_utils.py`: Defines a set of functions used for distributed training.
    - `inference.py`: Defines the evaluation loop and sliding window.
    - `logging.py`: Defines the MLPerf logger.
    - `training.py`: Defines the training loop.


### 准备数据集和预训练模型
下载数据集[KiTS19 github repository](https://github.com/neheller/kits19)
这将把原始的非插值数据下载到 `/data/raw-data-dir/kits19/data`,步骤如下：
  ```bash
    mkdir -p data/raw-data-dir
    cd data/raw-data-dir
    git clone https://github.com/neheller/kits19
    cd kits19
    pip3 install -r requirements.txt
    python3 -m starter_code.get_imaging
  ```
运行 `preprocess_dataset.py`
  mkdir -p /data/KiTS19/pre_data_dir
  ```bash
    python preprocess_dataset.py --data_dir /data/raw-data-dir/kits19/data --results_dir /data/KiTS19/pre_data_dir
  ```
`/data/KiTS19/pre_data_dir`目录下，包含的内容如下：
```
└── data
      ├── KiTS19
      │   ├── pre_data_dir
```
指定数据集和模型权重环境变量：
```bash
export PYTORCH_TRAIN_DATASET=/path/to/dataset

export PYTORCH_TRAIN_CHECKPOINT=/path/to/checkpoint
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
git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging
pip install -r requirements.txt
```
#### 使用Dockerfile 准备环境
1. 构建 docker 镜像
```bash
export IMAGE_NAME=demo_unet3d
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../
```
2. 创建并启动容器（指定数据集挂载目录）
```bash
docker run -it --ipc=host -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name mlu_unet3d --network=host $IMAGE_NAME
```
3. 配置容器环境
```bash
source env.sh
source /torch/venv3/pytorch/bin/activate
git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging
```
### 执行训练或推理脚本
```bash
bash run_scripts/Unet3d_FP32_4000E_4MLUs_Train.sh
```
## 一键训练脚本
| Models      | Framework | MLU       | Data Precision | Cards | Description                      | Run                                                         |
| ----------- | --------- | --------- | -------------- | ----- | -------------------------------- | ----------------------------------------------------------- |
| Unet3d      | PyTorch   | MLU370-X8 | FP32           | 4     | from scratch training use 4 MLUs | bash run_scripts/Unet3d_FP32_1000E_4MLUs_Train.sh  |
| Unet3d      | PyTorch   | MLU370-X8 | FP32           | 8     | from scratch training use 8 MLUs | bash run_scripts/Unt3d_FP32_1000E_8MLUs_Train.sh   |

## 一键推理脚本
执行推理脚本前需自行训练得到模型权重
| Models      | Framework | MLU       | Data Precision | Description                | Run                                          |
| ----------- | --------- | --------- | -------------- | -------------------------- | -------------------------------------------- |
| Unet3d      | PyTorch   | MLU370-S4 | FP32           | inference script           | bash run_scripts/Unet3d_Infer.sh      |
## 结果展示
Training accuracy result:MLU370-X8
| Models      | Epochs | FP32 mean_dice            |
| ----------- | ------ | --------------------------|
| Unet3d      | 4000   | 0.9076145887374878        |


## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

## Release_Notes
@TODO
