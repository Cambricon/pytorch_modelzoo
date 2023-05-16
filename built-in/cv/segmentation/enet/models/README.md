# ENet的PyTorch训练

本项目关于ENet模型的训练。

### 环境准备:

- Cambricon PyTorch 1.9.0

### 安装python依赖库

```
pip install -r requirements.txt
```

### Prepare Dataset Cityscapes

First, download the dataset from https://www.cityscapes-dataset.com/ into the data folder and then extract it.

### 设置环境变量指向Cityscapes数据集位置：

default env is export PYTORCH_TRAIN_DATASET=/data/pytorch/datasets/

and Cityscapes PATH is PYTORCH_TRAIN_DATASET/CityScapes/

### 8卡mlu from scratch训练

```
bash ./cambricon/scratch_enet_8mlu.sh
