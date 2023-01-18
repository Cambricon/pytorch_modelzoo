# SSD_ResNet50(PyTorch)
## **模型概述**
- 该Sample包含PYTORCH MODELZOO SSD_ResNet50的训练和推理的实现。
- SSD_ResNet50网络结构可参考GitHub链接：<https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD>。
- 基于COCO2017数据集训练脚本GitHub链接可参考：<https://github.com/NVIDIA/DeepLearningExamples/blob/437b950d5be2cc5d8044378ebf02976d5a21fc13/PyTorch/Detection/SSD/main.py>

## **支持情况**
### **训练模型支持情况**
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs |
----- | ----- | ----- | ----- | ----- | 
SSD_ResNet50  | PyTorch  | MLU370-X8  | FP32/AMP  | Yes  | 

### **推理模型支持情况**
Models  | Framework  | Supported MLU   | Supported Data Precision | Supported Infer Mode |
----- | ----- | ----- | ----- | ----- | 
SSD_ResNet50  | PyTorch  | MLU370-S4/X4  | FP32  | cnnl |

## 默认参数配置
### **模型训练默认参数配置**
以下为SSD_ResNet50模型的默认参数配置：
### Optimizer
模型默认优化器为SGD，以下为相关参数：
* Momentum: 0.9
* Learning Rate: 2.6e-3 for batch size 32
* Weight decay: 5e-4
* Epoch: 65

### **模型推理默认参数配置**
* backbone: 没有指定情况下，默认使用 torchvision 的预训练模型，可选指定训练完成的权重(eg. --modeldir /model/xxxx.pt)
* backbone-path: 32,64 (batch_size <= 64 in MLU370s4)
* input_data_type：默认使用 float32 可选float16
* bs：batch_size 默认32
* data:coco 数据集路径
* mode：evaluation模式即为推理模式
* checkpoint：推理checkpoint

## **依赖项检查**
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算版本MLU370-X8;
* Cambricon Driver >=v4.20.6；
* CNToolKit >=2.8.3;
* CNNL >=1.10.2;
* CNCL >=1.1.1;
* CNLight >=0.12.0;
* CNPyTorch >= 1.3.0;
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## **快速启动**
### **准备数据集**
该SSD_ResNet50脚本基于coco2017训练，数据集下载方式：
```
curl -O http://images.cocodataset.org/zips/train2017.zip; unzip train2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip; unzip val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip; unzip annotations_trainval2017.zip
```
数据集目录结构为：
```
├── annotations
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── images
│   ├── test2017 -> ../test2017/
│   ├── train2017 -> ../train2017
│   └── val2017 -> ../val2017/
├── test2017
├── train2017
├── train2017.txt
├── train2017.zip
├── val2017
├── val2017.shapes
└── val2017.txt
```

### **准备模型**
```
wget -c https://download.pytorch.org/models/resnet50-19c8e357.pth -O ${PYTORCH_TRAIN_CHECKPOINT}ssd/resnet50-19c8e357.pth
```

### **环境准备**
#### 基于base docker image安装
##### 1、导入镜像
```
##下载Cambricon PyTorch docker镜像
docker load -i xxx.tar.gz
```
##### 2、启动测试容器
```
bash run_docker.sh
```
##### 3、启动虚拟环境并安装依赖
```
#配置环境变量 激活虚拟环境 安装必要Python包
source env.sh
pip install -r models/requirements.txt
```
#### 使用Dockerfile准备环境(推荐)
#### 1、生成SSD_ResNet50的Docker镜像：
```
docker build --network=host -t modelzoo:ssd_resnet50 -f DOCKERFILE ../../../../
```

####  2、创建容器
```
docker run -it --network=host --ipc=host -v /data:/data  --device /dev/cambricon_ctl --privileged --name ssd_resnet50  modelzoo:ssd_resnet50
```

##### 3、启动虚拟环境并安装依赖
- 一键运行脚本
```
source env.sh
pip install -r models/requirements.txt
```
#### **一键执行训练脚本**
Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
SSD_ResNet50  | PyTorch1.6  | MLU370-X8  | AMP  | 4  | bash SSD_ResNet50_AMP_65E_4MLUs_Train.sh |
SSD_ResNet50  | PyTorch1.6  | MLU370-X8  | FP32  | 4  | bash SSD_ResNet50_FP32_65E_4MLUs_Train.sh |
#### **一键执行推理脚本**
Models  | Framework  | MLU   | Data Precision  |Run
----- | ----- | ----- | ----- | ----- |
SSD_ResNet50  | PyTorch1.6  | MLU370-S4  | FP32  | bash SSD_ResNet50_Infer.sh

## **结果展示**
- 无官方checkpoints 结果暂无
- 用户可基于SSD_ResNet50_AMP_65E_4MLUs_Train.sh 或者 SSD_ResNet50_FP32_65E_4MLUs_Train.sh训练得到的模型来进行推理，模型路径见脚本内部指定。
### **推理结果**


