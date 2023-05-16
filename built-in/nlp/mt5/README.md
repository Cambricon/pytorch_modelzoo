# MT5
---
## 模型概述
  | 模型名称 | MT5 |
| :---: | :--- |
| 论文 | [mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://arxiv.org/abs/2010.11934) |
| 模型实现| [transformers(huggingface)](https://github.com/huggingface/transformers)|
| 预训练模型 | [t5-pegasus](https://pan.baidu.com/s/15AHh2mm7nmlSd0TzdSI2Pw?pwd=1234) 预训练模型 |
| Fine-tune 数据集 | [CSL](https://github.com/ydli-ai/CSL) 数据集上的中文论文摘要 |
| 实现效果 | 用于中文论文摘要生成任务 |

## 支持情况
---
### 模型训练支持情况
| Models | Framework  | Supported MLU   | Supported Data Precision | Multi-Devices |
| ------ | ---------- | --------------- | ------------------------ | ------------- |
| MT5   | PyTorch1.6 && 1.9 | MLU370-X8/X4/S4 | FP32 or AMP              | Yes           |

### 模型推理支持情况
| Models | Framework  | Supported MLU   | Supported Data Precision |
| ------ | ---------- | --------------- | ------------------------ |
| MT5   | PyTorch1.6 && 1.9 | MLU370-X8/X4/S4 | FP32                     |


## 默认参数配置
---
### 模型训练默认参数

#### Optimizer
模型优化器为Adam，参数配置如下：
- Learning Rate: 2e-4 for batch size 16
- Epoch:4


## 环境依赖
---
- Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
- 服务器装配好寒武纪计算板卡MLU370-X8;
- Cambricon SDK 1.10
  - Cambricon Driver >=v4.20.18；
  - CNToolKit >=3.2.2;
  - CNNL >=1.15.2;
  - CNCL >=1.6.0;
  - CNLight >=0.18.0;
  - CNPyTorch >= 1.11.0;
- 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 快速使用指南
---
### 文件说明
- run_scripts/ 包含一键训练和推理的shell脚本文件
- models/ 包含模型文件
  - modeling_t5.py 对transformer库中同名文件的修改
  - finetune.py 模型训练脚本，更多信息使用`python finetune.py -h`查看
  - predict.py 模型推理验证脚本, 更多信息使用`python eval.py -h`查看
  - util.py 工具函数

### 准备数据集
下载数据集[CSL](https://rajpurkar.github.io/SQuAD-explorer/) 到当前目录
```
git clone https://github.com/ydli-ai/CSL
cd CSL
git reset --hard 7cb83f446ee0e5a00e3366e000cd5e086160e477
cd -
```

其中数据文件包括
```
CSL/benchmark/ts/dev.tsv
CSL/benchmark/ts/test.tsv
CSL/benchmark/ts/train.tsv
```

### 准备预训练模型
预训练模型下载链接：[百度云：chinese_t5_pegasus_base.zip](https://pan.baidu.com/s/15AHh2mm7nmlSd0TzdSI2Pw?pwd=1234), 提取码：1234并解压在当前文件夹，包含内容如下：

```
.
├── config.json
├── pytorch_model.bin
└── vocab.txt
```



### 基于base docker image安装

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
3. 配置容器环境(在容器内部)
```bash
source /torch/venv3/pytorch/bin/activate
pip install -r ./models/requirements.txt
cp models/modeling_t5.py $(dirname $(python -c "import transformers; print(transformers.__file__)"))/models/t5/modeling_t5.py
```

#### 使用Dockerfile 准备环境
1. 构建 docker 镜像
```bash
export IMAGE_NAME=demo_mt5
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../
```
2. 创建并启动容器（指定数据集挂载目录）
```bash
docker run -it --ipc=host -v /data:/data -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name mlu_mt5 --network=host $IMAGE_NAME
```

3. 配置容器环境(在容器内部)
```bash
source /torch/venv3/pytorch/bin/activate
```

### 执行训练或推理脚本

```bash
source env.sh
bash run_scripts/MT5_FP32_4E_4MLUs_Train.sh
```
## 一键训练脚本
| Models | Framework  | MLU             | Data Precision | Cards | Description                  | Run                                         |
| ------ | ---------- | --------------- | -------------- | ----- | ---------------------------- | ------------------------------------------- |
| MT5    | PyTorch1.6 && 1.9 | MLU370-X8/X4/S4 | FP32           | 4     | finetune training use 4 MLUs | bash run_scripts/MT5_FP32_4E_4MLUs_Train.sh |
| MT5    | PyTorch1.6 && 1.9 | MLU370-X8/X4/S4 | FP32           | 1     | finetune training use 1 MLUs | bash run_scripts/MT5_FP32_4E_1MLUs_Train.sh |
| MT5    | PyTorch1.6 && 1.9 | MLU370-X8/X4/S4 | AMP            | 4     | finetune training use 4 MLUs | bash run_scripts/MT5_AMP_4E_4MLUs_Train.sh  |
| MT5    | PyTorch1.6 && 1.9 | MLU370-X8/X4/S4 | AMP            | 1     | finetune training use 1 MLUs | bash run_scripts/MT5_AMP_4E_1MLUs_Train.sh  |

训练完成后，在当前目录下会生成`saved_model/summary_model`文件

## 一键推理脚本


| Models | Framework  | MLU             | Data Precision | Description      | Run                           |
| ------ | ---------- | --------------- | -------------- | ---------------- | ----------------------------- |
| MT5    | PyTorch1.6 && 1.9 | MLU370-X8/S4/X4 | FP32           | inference script | bash run_scripts/MT5_Infer.sh |


## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

数据集下载链接：[CSL](https://github.com/ydli-ai/CSL)
预训练模型下载链接：[chinese_t5_pegasus_base.zip](https://pan.baidu.com/s/15AHh2mm7nmlSd0TzdSI2Pw?pwd=1234)


## Release_Notes
@TODO
