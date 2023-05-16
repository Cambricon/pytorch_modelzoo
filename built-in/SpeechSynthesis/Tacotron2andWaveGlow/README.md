# Tacotron2(PyTorch)
## **模型概述**
Tacotron2是由Google Brain 2017年提出来的一个语音合成框架。
论文来源：https://arxiv.org/pdf/1712.05884.pdf
项目地址：https://github.com/NVIDIA/tacotron2
WaveGlow发表于 ICASSP 2019会议。
论文来源：https://arxiv.org/pdf/1811.00002v1.pdf
https://github.com/NVIDIA/waveglow

## **支持情况**
### **训练模型支持情况**
|   Models  |   Framework   |  Supported MLU  | Supported Data Precision  | Multi-GPUs  | 
| --------- | ------------- | --------------- | ------------------------- | ----------  | 
| Tacotron2 |   PyTorch1.6  |    MLU370-X8    |     FP32                  | Yes         |
| WaveGlow  |   PyTorch1.6  |    MLU370-X8    | AMP/FP32                  | Yes         |
### ** 推理模型支持情况**
|   Models  |   Framework   |  Supported MLU  | Supported Data Precision  |  Supported Infer Mode |
|-----------| ------------- | --------------- | ------------------------- |  -------------------- |
| Tacotron2 |   PyTorch1.6  |   MLU370-S4/X4  |     FP32                  |         CNNL          |
| WaveGlow  |   PyTorch1.6  |   MLU370-S4/X4  |     FP32                  |         CNNL          |
## 默认参数配置

### Optimizer
模型默认优化器为Adam，以下为相关参数：

* `--learning-rate` - 学习率 (Tacotron 2: 1e-3, WaveGlow: 1e-4)
* `--batch-size` - 批量大小 (Tacotron 2 FP16/FP32: 104/48, WaveGlow FP16/FP32: 10/4)
* `--Weight decay` - 权值衰减(Tacotron 2: 1e-6, WaveGlow: 0)
* `--grad_clip_thresh` - (Tacotron 2: 1.0, WaveGlow: 3.4028234663852886e+38)
* `--grad_clip` - (Tacotron 2: 5.0, WaveGlow: 0)
* `--epochs`- (Tacotron 2: 1501, WaveGlow: 1001)
* `--pyamp` - 使用混合精度训练

#### 共享音频/STFT 参数

* `--sampling-rate` - 输入和输出音频的采样率(22050)
* `--filter-length` - (1024)
* `--hop-length` - FFT的跳跃长度，即连续FFT之间的样本步幅 (256)
* `--win-length` - FFT的窗口大小 (1024)
* `--mel-fmin` - 以赫兹为单位的最低频率 (0.0)
* `--mel-fmax` - 以赫兹为单位的最低频率 (8.000)

#### Tacotron 2 参数

* `--anneal-steps` - 退火学习率的时期 (500 1000 1500)
* `--anneal-factor` - 退火学习率的因子(FP16/FP32: 0.3/0.1)

#### WaveGlow 参数

* `--segment-length` - 神经网络处理的输入音频的段长度 (8000)
* `--wn-channels` - 耦合层网络中的剩余通道数 (512)


## **依赖项检查**
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算板卡MLU370-X8;
* Cambricon Driver >=v4.20.6；
* CNToolKit >=2.8.3;
* CNNL >=1.10.2;
* CNCL >=1.1.1;
* CNLight >=0.12.0;
* CNPyTorch >= 1.3.0;
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO
## **快速使用指南**
### **文件说明**
* run_scripts/ 包含一键训练和推理的shell脚本文件
* models/ 包含原始模型仓库文件
 。data:载入和打包训练数据以及数据集下载脚本
 。inference.py:模型推理脚本
 。train.py:模型训练脚本，更多信息使用python train.py -h查看
### **数据集准备**
1.下载数据集https://keithito.com/LJ-Speech-Dataset/ ,并解压，包含的内容如下：
目录结构：
```
├── mels
│   ├── LJ001-0001.pt
│   ├── LJ001-0002.pt
│   ├── ...
├── wavs
│   ├── LJ001-0001.wav
│   ├── LJ001-0002.wav
│   ├── ...
├── metadata.csv
├── README
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
source /torch/venv3/pytorch/bin/activate
pip install -r ./models/requirements.txt
```
#### 使用Dockerfile 准备环境
1. 构建 docker 镜像
```bash
export IMAGE_NAME=demo_tacotron2
docker build --network=host -t $IMAGE_NAME -f DOCKERFILE ../../../
```
2. 创建并启动容器（请自行指定数据集和权重挂载目录）
```bash
docker run -it --ipc=host -v /data:/data  --device /dev/cambricon_ctl --privileged --name tacotron2 --network=host $IMAGE_NAME
```
3. 配置容器环境
```bash
source env.sh
source /torch/venv3/pytorch/bin/activate
```

### 执行训练或推理脚本
bash run_scripts/Tacotron2/Tacotron2_FP32_1501E_4MLUs_Train.sh
bash run_scripts/WaveGlow/Waveglow_AMP_1001E_4MLUs_Train.sh
## 一键训练脚本
| Models      | Framework | MLU       | Data Precision | Cards |   Run                                                         | 
| ----------- | --------- | --------- | -------------- | ----- |---------------------------------------------------------------|
| Tacotron2   | PyTorch   | MLU370-X8 | FP32           | 4     |   bash run_scripts/
Tacotron2_FP32_1501E_4MLUs_Train.sh   |

| Models      | Framework | MLU       | Data Precision | Cards |   Run                                                         | 
| ----------- | --------- | --------- | -------------- | ----- |---------------------------------------------------------------|
| WaveGlow   | PyTorch   | MLU370-X8 | FP32           | 4     |   bash run_scripts/
Waveglow_AMP_1001E_1MLUs_Train.sh   |
| WaveGlow   | PyTorch   | MLU370-X8 | AMP            | 4     |   bash run_scripts/
Waveglow_AMP_1001E_4MLUs_Train.sh    |


## 一键推理脚本
执行推理脚本前需自行训练得到模型权重
| Models      | Framework | MLU       | Data Precision | Description                | Run                                          |
| ----------- | --------- | --------- | -------------- | -------------------------- | -------------------------------------------- |
| Tacotron2   | PyTorch   | MLU370-S4 | FP32           | inference script           | bash run_scripts/
Tacotron2_Infer.sh |

| Waveglow    | PyTorch   | MLU370-S4 | FP32           | inference script           | bash run_scripts/
Waveglow_Infer.sh  |



   命令行选项运行
   训练所有可选参数如下：
   python train.py  -h

   usage: train.py [-h] -o OUTPUT [-d DATASET_PATH] -m MODEL_NAME
                [--log-file LOG_FILE]
                [--anneal-steps [ANNEAL_STEPS [ANNEAL_STEPS ...]]]
                [--anneal-factor {0.1,0.3}] [--config-file CONFIG_FILE]
                [--use-mlu] [--seed SEED] --epochs EPOCHS [--iter ITER]
                [--epochs-per-checkpoint EPOCHS_PER_CHECKPOINT]
                [--checkpoint-path CHECKPOINT_PATH] [--resume-multi-device]
                [--resume-from-last]
                [--dynamic-loss-scaling DYNAMIC_LOSS_SCALING] [--pyamp]
                [--cudnn-enabled] [--cudnn-deterministic]
                [--disable-uniform-initialize-bn-weight]
                [--use-saved-learning-rate USE_SAVED_LEARNING_RATE] -lr
                LEARNING_RATE [--weight-decay WEIGHT_DECAY]
                [--grad-clip-thresh GRAD_CLIP_THRESH] -bs BATCH_SIZE
                [--grad-clip GRAD_CLIP] [--load-mel-from-disk]
                [--training-files TRAINING_FILES]
                [--validation-files VALIDATION_FILES]
                [--text-cleaners [TEXT_CLEANERS [TEXT_CLEANERS ...]]]
                [--max-wav-value MAX_WAV_VALUE]
                [--sampling-rate SAMPLING_RATE]
                [--filter-length FILTER_LENGTH] [--hop-length HOP_LENGTH]
                [--win-length WIN_LENGTH] [--mel-fmin MEL_FMIN]
                [--mel-fmax MEL_FMAX] [--rank RANK] [--world-size WORLD_SIZE]
                [--dist-url DIST_URL] [--group-name GROUP_NAME]
                [--dist-backend {nccl,cncl}] [--bench-class BENCH_CLASS]

PyTorch Tacotron 2 Training

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Directory to save checkpoints
  -d DATASET_PATH, --dataset-path DATASET_PATH
                        Path to dataset
  -m MODEL_NAME, --model-name MODEL_NAME
                        Model to train
  --log-file LOG_FILE   Filename for logging
  --anneal-steps [ANNEAL_STEPS [ANNEAL_STEPS ...]]
                        Epochs after which decrease learning rate
  --anneal-factor {0.1,0.3}
                        Factor for annealing learning rate
  --config-file CONFIG_FILE
                        Path to configuration file
  --use-mlu             Enable MLU
  --seed SEED           manually set random seed for torch

training setup:
  --epochs EPOCHS       Number of total epochs to run
  --iter ITER           Number of total epochs to run
  --epochs-per-checkpoint EPOCHS_PER_CHECKPOINT
                        Number of epochs per checkpoint
  --checkpoint-path CHECKPOINT_PATH
                        Checkpoint path to resume training
  --resume-multi-device
                        Resumes training from the last multidevice checkpoint.
  --resume-from-last    Resumes training from the last checkpoint; uses the
                        directory provided with '--output' option to search
                        for the checkpoint "checkpoint_<model_name>_last.pt"
  --dynamic-loss-scaling DYNAMIC_LOSS_SCALING
                        Enable dynamic loss scaling
  --pyamp               Enable PYAMP
  --cudnn-enabled       Enable cudnn
  --cudnn-deterministic
                        Run cudnn deterministic
  --disable-uniform-initialize-bn-weight
                        disable uniform initialization of batchnorm layer
                        weight

optimization setup:
  --use-saved-learning-rate USE_SAVED_LEARNING_RATE
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learing rate
  --weight-decay WEIGHT_DECAY
                        Weight decay
  --grad-clip-thresh GRAD_CLIP_THRESH
                        Clip threshold for gradients
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size per GPU
  --grad-clip GRAD_CLIP
                        Enables gradient clipping and sets maximum gradient
                        norm value

dataset parameters:
  --load-mel-from-disk  Loads mel spectrograms from disk instead of computing
                        them on the fly
  --training-files TRAINING_FILES
                        Path to training filelist
  --validation-files VALIDATION_FILES
                        Path to validation filelist
  --text-cleaners [TEXT_CLEANERS [TEXT_CLEANERS ...]]
                        Type of text cleaners for input text

audio parameters:
  --max-wav-value MAX_WAV_VALUE
                        Maximum audiowave value
  --sampling-rate SAMPLING_RATE
                        Sampling rate
  --filter-length FILTER_LENGTH
                        Filter length
  --hop-length HOP_LENGTH
                        Hop (stride) length
  --win-length WIN_LENGTH
                        Window length
  --mel-fmin MEL_FMIN   Minimum mel frequency
  --mel-fmax MEL_FMAX   Maximum mel frequency

distributed setup:
  --rank RANK           Rank of the process, do not set! Done by multiproc
                        module
  --world-size WORLD_SIZE
                        Number of processes, do not set! Done by multiproc
                        module
  --dist-url DIST_URL   Url used to set up distributed training
  --group-name GROUP_NAME
                        Distributed group name
  --dist-backend {nccl,cncl}
                        Distributed run backend

benchmark:
  --bench-class BENCH_CLASS

** 推理所有可选参数如下：**
python inference.py  -h
usage: inference.py [-h] -i INPUT -o OUTPUT [--suffix SUFFIX]
                    [--tacotron2 TACOTRON2] [--waveglow WAVEGLOW]
                    [-s SIGMA_INFER] [-d DENOISING_STRENGTH]
                    [-sr SAMPLING_RATE] [--fp16] [--device DEVICE]
                    [--log-file LOG_FILE] [--include-warmup]
                    [--stft-hop-length STFT_HOP_LENGTH]

PyTorch Tacotron 2 Inference

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        full path to the input text (phareses separated by new
                        line)
  -o OUTPUT, --output OUTPUT
                        output folder to save audio (file per phrase)
  --suffix SUFFIX       output filename suffix
  --tacotron2 TACOTRON2
                        full path to the Tacotron2 model checkpoint file
  --waveglow WAVEGLOW   full path to the WaveGlow model checkpoint file
  -s SIGMA_INFER, --sigma-infer SIGMA_INFER
  -d DENOISING_STRENGTH, --denoising-strength DENOISING_STRENGTH
  -sr SAMPLING_RATE, --sampling-rate SAMPLING_RATE
                        Sampling rate
  --fp16                Run inference with mixed precision
  --device DEVICE       Run inference on GPU,MLU or CPU
  --log-file LOG_FILE   Filename for logging
  --include-warmup      Include warmup
  --stft-hop-length STFT_HOP_LENGTH
                        STFT hop length for estimating audio length from mel
                        size

## **结果展示**
训练结果
Training accuracy results: MLU370-X8

** 推理结果**
| Models      | MLUs |  val_loss           | 
| ----------- | ---- | ------------------  | 
| Tacotron2   |   4  |  2.6804521083831787 | 
| WaveGlow    |   4  |  -2.939938988004412 | 

## 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
公开的数据集地址：https://keithito.com/LJ-Speech-Dataset/ 
## Release_Notes
 @TODO         
