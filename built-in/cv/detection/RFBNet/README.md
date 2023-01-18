# RFBNet(PyTorch)
## **模型概述**

- 该Sample包含PYTORCH MODELZOO RFBNet的训练和推理的实现。
- RFBNet网络结构可参考GitHub链接：<https://github.com/ruinmessi/RFBNet/>。
- 基于VOC数据集训练脚本GitHub链接可参考：<https://github.com/GOATmessi7/RFBNet/blob/master/train_RFB.py>

## **支持情况**
### **训练模型支持情况**
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  |
----- | ----- | ----- | ----- | ----- | 
RFBNet  | PyTorch  | MLU370-X8  | FP32/AMP  | Yes  | 

### **推理模型支持情况**
Models  | Framework  | Supported MLU   | Supported Data Precision  
----- | ----- | ----- | ----- | 
RFBNet  | PyTorch  | MLU370-S4/X4  | FP32  |

## 默认参数配置
### **模型训练默认参数配置**
以下为RFBNet模型的默认参数配置：
### Optimizer
模型默认优化器为SGD，以下为相关参数：
* Momentum: 0.9
* Learning Rate: 4e-3 for batch size 32
* Weight decay: 5e-4
* epoch: 300

### **模型推理默认参数配置**
* version:RFB_vgg 网络结构
* size:300 图像尺寸
* trained_model:训练模型路径
* save_folder: 推理结果保存路径
* cpu:是否使用cpu nms
* device:推理使用的设备 默认mlu

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
该RFBNet脚本基于VOC2007和VOC2012训练，可参见models/data/scripts/VOC2007.sh进行下载:
数据集目录结构为：
```
├── Annotations
│   ├── xxx.xml
├── ImageSets
│   ├── Layout
│   ├── Main
│   └── Segmentation
├── JPEGImages
│   ├──xxx.jpg
├── SegmentationClass
│   ├──xxx.png
├── SegmentationObject
│   ├──xxx.png
├── annotations_cache
├── log
├── mobilenetssd_file_list
```

### **准备模型**
```
wget -c https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth -O ${PYTORCH_TRAIN_CHECKPOINT}rfbnet/checkpoints_fp/vgg16_reducedfc.pth
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

#### 使用Dockerfile准备环境
#### 1、生成RFBNet的Docker镜像：
```
docker build --network=host -t modelzoo:rfbnet -f DOCKERFILE ../../../../
```
####  2、创建容器
```
docker run -it --network=host --ipc=host -v /data:/data  --device /dev/cambricon_ctl --privileged --name rfbnet  modelzoo:rfbnet /bin/bash
```

##### 3、启动虚拟环境并安装依赖
```
#配置环境变量 激活虚拟环境 安装必要Python包
source env.sh
pip install -r models/requirements.txt
```

#### **一键执行训练脚本**
Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
RFBNet  | PyTorch  | MLU370-X8  | AMP   | 4  | bash  RFBNet_AMP_300E_4MLUs_Train.sh
RFBNet  | PyTorch  | MLU370-X8  | FP32  | 4  | bash  RFBNet_FP32_300E_4MLUs_Train.sh


#### **一键执行推理脚本**
Models  | Framework  | MLU   | Data Precision  |Run
----- | ----- | ----- | ----- | ----- | 
RFBNet  | PyTorch  | MLU370-S4  | FP32  | bash RFBNet_Infer.sh

## **结果展示**
### **推理结果**
##### **Accuracy: MLU370-X4**
- Final_RFB_vgg_VOC.pth基于RFBNet_AMP_300E_4MLUs_Train.sh训练得到，路径见脚本内部指定。

Models  | Checkpoints  | Batch_Size | Precision | Cards | mAP |
----- | ----- | ----- | ----- | ----- | ----- |
RFBNet | Final_RFB_vgg_VOC.pth  | 32 | FP32 | 0.8055 | 
