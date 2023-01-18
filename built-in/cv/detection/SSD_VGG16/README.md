# SSD_VGG16(PyTorch)
## **模型概述**
- 该Sample包含PYTORCH MODELZOO SSD_VGG16基于VOC2007的训练和推理的实现,用户可自行拓展使用VOC2012和COCO数据集。
- SSD_VGG16网络结构可参考GitHub链接：<https://github.com/amdegroot/ssd.pytorch>。
- 基于VOC数据集训练脚本GitHub链接可参考：<https://github.com/amdegroot/ssd.pytorch/blob/master/train.py>
## **支持情况**
### **训练模型支持情况**
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  |
----- | ----- | ----- | ----- | ----- | 
SSD_VGG16  | PyTorch  | MLU370-X8  | FP32/AMP  | Yes  
### **推理模型支持情况**
Models  | Framework  | Supported MLU   | Supported Data Precision  
----- | ----- | ----- | ----- | 
SSD_VGG16  | PyTorch  | MLU370-S4/X4  | FP32  |

## 默认参数配置
### **模型训练默认参数配置**
以下为SSD_VGG16模型的默认参数配置：
### Optimizer
模型默认优化器为SGD，以下为相关参数：
* Momentum: 0.9
* Learning Rate: 2e-3 for batch size 64
* Weight decay: 5e-4
* Iterations: 60000

### **模型推理默认参数配置**
* backbone: 没有指定情况下，默认使用 torchvision 的预训练模型，可选指定训练完成的权重(eg. --modeldir /model/xxxx.pt)
* backbone-path: 32,64 (batch_size <= 64 in MLU370s4)
* input_data_type：默认使用 float32 可选float16
* bs：batch_size 默认32
* data:coco 数据集路径
* mode：evaluation模式即为推理模式
* checkpoint：推理采用的checkpoint

# Eval model After Train based on last iteration
* trained_model:训练模型权重文件
* voc_root:voc数据集路径
* device: 推理设备 "mlu" or "gpu"

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

## **Quick Start Guide**
### **数据集准备**  (说明文件修改的地方)
该SSD_VGG16脚本基于VOC2007训练，可参见models/data/scripts/VOC2007.sh进行下载:
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

wget -c https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth -O $PROJ_DIR/data/weights/gpu_checkpoints/ssd300_mAP_77.43_v2.pth
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
#### 1、生成SSD_VGG16的Docker镜像：
```
docker build --network=host -t modelzoo:ssd_vgg16 -f DOCKERFILE ../../../../
```

####  2、创建并启动容器
```
docker run -it --network=host --ipc=host -v /data:/data  --device /dev/cambricon_ctl --privileged --name ssd_vgg16  modelzoo:ssd_vgg16
```

##### 3、启动虚拟环境并安装依赖
- 请在env.sh当中设置DATASETS_PATH和PROJ_DIR
```
source env.sh
pip install -r models/requirements.txt
```

#### **一键执行训练脚本**
Models  | Framework  | MLU   | Data Precision  | Description |Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |----- |
SSD_VGG16  | PyTorch  | MLU370-X8  | AMP | 4MLUsTrain | 4  | bash SSD_VGG16_AMP_60000S_4MLUs_Train.sh |
SSD_VGG16  | PyTorch  | MLU370-X8  | FP32    | 4MLUsTrain | 4  | bash SSD_VGG16_FP32_60000S_4MLUs_Train.sh |


#### **一键执行推理脚本**
- 执行推理脚本前需自行得到模型权重

Models  | Framework  | MLU   | Data Precision  | Description | Run
----- | ----- | ----- | ----- | ----- | ----- |
SSD_VGG16  | PyTorch  | MLU370-S4  | FP32  | Eval Mode | bash SSD_VGG16_Infer.sh |

## **结果展示**
### **推理结果**
##### **Accuracy: MLU370-X4**
- 使用的模型为官方checkpoints,获取方式见上方准备模型章节。

Models  | Checkpoints  | Precision | Cards | mAP| 
----- | ----- | ----- | ----- | ----- | 
SSD_VGG16 | ssd300_mAP_77.43_v2.pth | FP32 | 1 | 0.7749 |


