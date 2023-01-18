# Bert_Base_Finetune_Msra_Ner(Pytorch)
---
## 模型概述
  本仓库为Bert_Base_Finetune_Msra_Ner的MLU实现，利用Google AI的BERT模型进行中文命名实体识别任务。原始GPU实现仓库为: [NER-BERT-pytorch](https://github.com/lemonhu/NER-BERT-pytorch)。

## 支持情况
---
### 训练模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-MLUs |
----- | ----- | ----- | ----- | ----- |
Bert_Base_Finetune_Msra_Ner  | PyTorch1.6  | MLU370-X8  | AMP/FP32  | Yes  |

### 推理模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Supported Infer Mode | 
----- | ----- | ----- | ----- | ----- |
Bert_Base_Finetune_Msra_Ner  | PyTorch1.6  | MLU370-X8  | FP16/FP32      | CNNL |

## 默认参数配置
---
### Optimizer
Models  | 优化器  | Learning Rate   | Learning rate schedule |  Weight decay | Epoch
---- | ----- | ----- | ----- | ----- | ----- |
Bert_Base_Finetune_Msra_Ner  | Adam  | 3e-05  | LambdaLR  | 0 | 20

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
- `train_ddp.py` 模型训练入口，更多信息使用`python train_ddp.py -h`查看
- `evaluate.py` 模型推理入口，更多信息使用`python evaluate.py -h`查看

### 准备数据集
在路径 bert_base_finetune_msra_ner/models/data/msra 下，包含原始数据集文件 `msra_train_bio` 和 `msra_test_bio`。

运行 build_msra_dataset_tags.py 生成 `train/val/test` 数据集和包含 tags 的 `tags.txt`。 `val` 中的数据集由 `msra_train_bio` 切分而来。(也可以直接使用`data/msra/`下已生成好的数据。)

```shell
python build_msra_dataset_tags.py
```

数据集目录最终排布如下：

 ```bash 
./data/msra/
├── msra_test_bio
├── msra_train_bio
├── tags.txt
├── test
│   ├── sentences.txt
│   └── tags.txt
├── train
│   ├── sentences.txt
│   └── tags.txt
└── val
    ├── sentences.txt
    └── tags.txt
  ```

### 准备预训练 BERT 权重
需要手动从 Tensorflow checkpoint 转换成 PyTorch 的dump文件(建议在下一节的docker环境中完成该操作)。

- 下载并解压 [BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) (Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters).

- 执行下述命令，将Tensorflow checkpoint 转换成 PyTorch 的dump文件。

  ```bash
  pip install tensorflow-cpu
  pytorch_pretrained_bert==0.4.0
  export TF_BERT_BASE_DIR=/path/to/chinese_L-12_H-768_A-12
  export PT_BERT_BASE_DIR=/path/to/models/bert-base-chinese-pytorch
  
  pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch \
  $TF_BERT_BASE_DIR/bert_model.ckpt \
  $TF_BERT_BASE_DIR/bert_config.json \
  $PT_BERT_BASE_DIR/pytorch_model.bin
  ```

- 拷贝 BERT 参数文件 `bert_config.json` 和字典文件 `vocab.txt` 到目录 `$PT_BERT_BASE_DIR`.

   ```bash
   cp $TF_BERT_BASE_DIR/bert_config.json $PT_BERT_BASE_DIR/bert_config.json
   cp $TF_BERT_BASE_DIR/vocab.txt $PT_BERT_BASE_DIR/vocab.txt
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
## 默认的 MY_CONTAINER 已设置为 bert_base_finetune_msra_ner_pytorch_1_6_0

bash run_docker.sh
```

##### 3、在容器中设置环境变量、安装依赖

```bash
## env.sh中的`PYTORCH_TRAIN_CHECKPOINT`为容器内 BERT 预训练权重路径，
## 这个环境变量需要用户根据真实情况设置。

source env.sh
pip install -r models/requirements.txt
```

#### 使用Dockerfile准备环境
##### 1、构建 docker 镜像
```bash
export IMAGE_NAME=test_bert_base_finetune_msra_ner_pytorch_1_6_0
## ../../../路径下包含tools/  built-in/ 等文件夹。
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../
```

##### 2、创建并启动容器

```bash
## 注意：默认的容器名 name 已设置为 test_mlu_bert_base_finetune_msra_ner

docker run -it --ipc=host -v /usr/bin/cnmon:/usr/bin/cnmon --device /dev/cambricon_ctl --privileged --name test_mlu_bert_base_finetune_msra_ner --network=host $IMAGE_NAME
```

##### 3、在容器中设置环境变量、安装依赖

```bash
## env.sh中的`PYTORCH_TRAIN_CHECKPOINT`为容器内 BERT 预训练权重路径
## 这个环境变量需要用户根据真实情况设置。

source env.sh
pip install -r models/requirements.txt
```

### 执行训练或推理脚本
```bash
bash run_scripts/Bert_Base_Finetune_Msra_Ner_FP32_20E_4MLUs_Train.sh
```

## 一键训练脚本
---
注意：训练过程中，生成的权重和log默认保存在 models/experiments/base_model 路径下，由 --model_dir 指定。
| Models      | Framework | MLU       | Data Precision | Cards | Description                      | Run                                                         |
| ----------- | --------- | --------- | -------------- | ----- | -------------------------------- | ----------------------------------------------------------- |
| Bert_Base_Finetune_Msra_Ner | PyTorch1.6| MLU370-X8 | FP32          | 4     | training use 4 MLUs  | bash run_scripts/Bert_Base_Finetune_Msra_Ner_FP32_20E_4MLUs_Train.sh  |
| Bert_Base_Finetune_Msra_Ner | PyTorch1.6| MLU370-X8 | AMP           | 4     | training use 4 MLUs  | bash run_scripts/Bert_Base_Finetune_Msra_Ner_AMP_20E_4MLUs_Train.sh   |

## 一键推理脚本
---
注意: 默认使用的推理权重为 --model_dir 和 --restore_file 共同指定的 experiments/base_model/best.pth.tar。

| Models      | Framework | MLU       | Data Precision | Description                | Run                                          |
| ----------- | --------- | --------- | -------------- | -------------------------- | -------------------------------------------- |
| Bert_Base_Finetune_Msra_Ner | PyTorch1.6  | MLU370-X8 | FP32           | inference script           | bash run_scripts/Bert_Base_Finetune_Msra_Ner_Infer.sh |


## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

## Release_Notes
@TODO
