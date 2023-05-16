# MaskRCNN(PyTorch)
## **模型概述**

- 该Sample包含PYTORCH MODELZOO MaskRCNN的训练和推理的实现。
- MaskRCNN网络结构可参考GitHub链接：<https://github.com/facebookresearch/maskrcnn-benchmark>。
- 基于COCO2017数据集训练脚本GitHub链接可参考：<https://github.com/facebookresearch/maskrcnn-benchmark/tools/train_net.py>


## **支持情况**
### **训练模型支持情况**
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  |
----- | ----- | ----- | ----- | ----- | 
MaskRCNN  | PyTorch  | MLU370-X8  | FP32  | Yes  | 

### **推理模型支持情况**
Models  | Framework  | Supported MLU   | Supported Data Precision  
----- | ----- | ----- | ----- | 
MaskRCNN  | PyTorch  | MLU370-S4/X4  | FP32  |

## 默认参数配置
### **模型训练默认参数配置**
以下为MaskRCNN模型的默认参数配置：
### Optimizer
模型默认优化器为SGD，以下为相关参数：
* Momentum: 0.9
* Learning Rate: 2e-2 for batch size 16
* Weight decay: 1e-4
* max_iter: 90000

### Data Augmentation
模型使用了以下数据增强方法：
* 训练
    * Cropping 300*300
    * Resize 300*300
    * Flipping
    * Jittering
* 验证
    * Cropping 300*300
    * Resize 300*300
    * Flipping
    * Jittering

### ** 模型推理默认参数配置**
* backbone: 没有指定情况下，默认使用 torchvision 的预训练模型，可选指定训练完成的权重(eg. --modeldir /model/xxxx.pt)
* backbone-path: 32,64 (batch_size <= 64 in MLU370s4)
* input_data_type：默认使用 float32 可选float16
* bs：batch_size 默认32
* data:coco 数据集路径
* mode：evaluation模式即为推理模式
* checkpoint：推理采用的checkpoint

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
该MaskRCNN脚本基于coco2017训练，数据集下载方式：
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
wget -c https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-101.pkl -O ${PYTORCH_TRAIN_CHECKPOINT}rcnn/basenet/R-101.pkl

wget -c https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_101_FPN_1x.pth -O $PROJ_DIR/data/weights/gpu_checkpoints/e2e_mask_rcnn_R_101_FPN_1x.pth
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
cd models && python setup.py build develop
```

#### 使用Dockerfile准备环境
#### 1、生成MaskRCNN的Docker镜像：
```
docker build --network=host -t modelzoo:maskrcnn -f DOCKERFILE ../../../../
```

####  2、创建容器
```
docker run -it --network=host --ipc=host -v /data:/data  --device /dev/cambricon_ctl --privileged --name maskrcnn  modelzoo:maskrcnn /bin/bash
```

##### 3、启动虚拟环境并安装依赖
```
#配置环境变量 激活虚拟环境 安装必要Python包
source env.sh
pip install -r models/requirements.txt
cd models && python setup.py build develop
```

#### **一键执行训练脚本**
Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
MaskRCNN  | PyTorch  | MLU370-X8  | FP32  | 8  | bash MaskRCNN_FP32_20000S_8MLUs_Train.sh


#### **一键执行推理脚本**
Models  | Framework  | MLU   | Data Precision  |Run |
----- | ----- | ----- | ----- | ----- | 
MaskRCNN  | PyTorch  | MLU370-S4  | FP32  | bash MaskRCNN_Infer.sh

## **结果展示**
### **推理结果**
##### **Accuracy: MLU370-X4**
- e2e_faster_rcnn_R_101_FPN_1x.pth获取方式见准备模型章节。

Models  | Checkpoints  | Batch_Size | Precision | Cards | IoU=0.50:0.95 | IoU=0.50 |
----- | ----- | ----- | ----- | ----- | ----- |----- |
MaskRCNN | e2e_mask_rcnn_R_101_FPN_1x.pth  | 32 | FP32 | 1 | 0.359 | 0.575 |

