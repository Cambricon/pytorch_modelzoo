# PyTorch ModelZoo 

## 介绍

PyTorch是时下最流行的AI框架，寒武纪对其进行了定制化开发，新增了对寒武纪加速板卡及寒武纪AI软件栈的支持，通常称之为Cambricon PyTorch。相比于原生PyTorch，用户基本不用做任何代码改动即可快速地将AI模型迁移至Cambricon PyTorch上。

针对CV 分类、检测、分割、NLP、语音等场景常用的各类经典和前沿的AI模型，本仓库展示了如何对其进行适配，使其可运行在Cambricon PyTorch上。开发者在进行其他AI 应用移植时可参考本仓库。


## 网络支持列表和链接

CV：

| MODELS | FRAMEWORK | Train Mode |Distributed Train| Infer  Mode
| ------------- | ------------- | ------------- | ------------- | ------------- | 
| [ResNet50](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32&&AMP|YES| TorchMM/CNNL |
| [ResNet18](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32|YES| TorchMM/CNNL | 
| [VGG16](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32&&AMP|YES| TorchMM/CNNL | 
| [MobileNetv2](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32&&AMP|YES| TorchMM/CNNL | 
| [AlexNet](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32|YES| TorchMM/CNNL | 
| [GoogLeNet](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32&&AMP|YES| TorchMM/CNNL | 
| [ResNet101](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32|YES| TorchMM/CNNL | 
| [VGG19](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32|YES| TorchMM/CNNL | 
| [VGG16_bn](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32|YES| TorchMM/CNNL |
| [ShuffleNet_v2_x0_5](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32|YES| TorchMM/CNNL |
| [ShuffleNet_v2_x1_0](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32|YES| TorchMM/CNNL |
| [ShuffleNet_v2_x1_5](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32|YES| TorchMM/CNNL |
| [ngc-resnet50v1_5](built-in/cv/classification/ngc-resnet50v1_5) | PyTorch1.6|FP32&&AMP|YES| TorchMM/CNNL |
| [Inceptionv2](built-in/cv/classification/Inceptionv2) | PyTorch1.6|FP32&&AMP|YES| TorchMM/CNNL |
| [Inceptionv3](built-in/cv/classification/timm) | PyTorch1.6|FP32&&AMP|YES| TorchMM/CNNL |
| [Inceptionv4](built-in/cv/classification/timm) | PyTorch1.9|FP32&&AMP|YES| TorchMM/CNNL |
| [OLTR](built-in/cv/classification/OLTR) | PyTorch1.6|FP32|NO| TorchMM/CNNL |
| [P3D](built-in/cv/classification/P3D) | PyTorch1.6|FP32|FP32&&AMP| TorchMM/CNNL |
| [Swin-Transformer-SSL](built-in/cv/classification/Swin-Transformer-SSL) | PyTorch1.9|FP32&&AMP|YES| TorchMM/CNNL |
| [swin_transformer](built-in/cv/classification/swin_transformer) | PyTorch1.6|AMP|YES| TorchMM/CNNL |
| [crnn](built-in/cv/classification/crnn) | PyTorch1.6|FP32|YES| TorchMM/CNNL |
| [FasterRCNN](built-in/cv/detection/FasterRCNN) | PyTorch1.6|FP32|YES| TorchMM/CNNL |
| [MaskRCNN](built-in/cv/detection/MaskRCNN) | PyTorch1.6|FP32|YES| TorchMM/CNNL |
| [RFBNet](built-in/cv/detection/RFBNet) | PyTorch1.6|FP32|FP32&&AMP| TorchMM/CNNL |
| [SSD_ResNet50](built-in/cv/detection/SSD_ResNet50) | PyTorch1.6|FP32&&AMP|YES| TorchMM/CNNL |
| [SSD_VGG16](built-in/cv/detection/SSD_VGG16) | PyTorch1.6|FP32&&AMP|YES| TorchMM/CNNL |
| [yolov3](built-in/cv/detection/yolov3) | PyTorch1.6|FP32&&AMP|YES| TorchMM/CNNL |
| [yolov5s](built-in/cv/detection/yolov5s) | PyTorch1.6|FP32&&AMP|YES| TorchMM/CNNL |
| [PointPillar](built-in/cv/detection/PointPillar) | PyTorch1.6|FP32|YES| TorchMM/CNNL |
| [unet3d](built-in/cv/segmentation/unet3d) | PyTorch1.6|FP32|YES| TorchMM/CNNL |
| [CycleGAN_and_pix2pix](built-in/cv/GAN/CycleGAN_and_pix2pix) | PyTorch1.6|FP32|NO| TorchMM/CNNL |

NLP:

| MODELS | FRAMEWORK | Train Mode |Distributed Train| Infer  Mode
| ------------- | ------------- | ------------- | ------------- | ------------- | 
| [transformer](built-in/nlp/transformer) | PyTorch1.6|FP32|YES| CNNL |
| [BERT](built-in/nlp/BERT/) | PyTorch1.6|FP32|YES| CNNL |
| [bert-base-cased](built-in/nlp/bert-base-cased) | PyTorch1.6|FP32&&AMP|YES| CNNL |
| [CRF](built-in/nlp/CRF) | PyTorch1.6|FP32|YES| CNNL |
| [bert_base_finetune_msra_ner](built-in/nlp/bert_base_finetune_msra_ner) | PyTorch1.6|FP32&&AMP|YES| CNNL |
| [DeepSpeech2](built-in/nlp/DeepSpeech2) | PyTorch1.6|FP32|YES| CNNL |

SpeechSynthesis/:

| MODELS | FRAMEWORK | Train Mode |Distributed Train| Infer  Mode
| ------------- | ------------- | ------------- | ------------- | ------------- | 
| [Tacotron2](built-in/SpeechSynthesis/Tacotron2andWaveGlow) | PyTorch1.6|FP32|YES| CNNL |
| [WaveGlow](built-in/SpeechSynthesis/Tacotron2andWaveGlow) | PyTorch1.6|AMP|YES| CNNL |
| [WaveRNN](built-in/SpeechSynthesis/WaveRNN) | PyTorch1.6|FP32&&AMP|YES| CNNL |

recommendation/:

| MODELS | FRAMEWORK | Train Mode |Distributed Train| Infer  Mode
| ------------- | ------------- | ------------- | ------------- | ------------- | 
| [DLRM](built-in/recommendation/DLRM) | PyTorch1.6|FP32&&AMP|YES| CNNL |


## issues/wiki/forum 跳转链接

## contrib 指引和链接

## LICENSE

PyTorch ModelZoo  的 License 具体内容请参见[LICENSE](LICENSE)文件。

## 免责声明

PyTorch ModelZoo 仅提供公共数据集以及预训练模型的下载链接，公共数据集及预训练模型并不属于 PyTorch ModelZoo ，PyTorch ModelZoo  也不对其质量或维护承担责任。请您在使用公共数据集和预训练模型的过程中，确保符合其对应的使用许可。

如果您不希望您的数据集或模型公布在 PyTorch ModelZoo上，或者您希望更新 PyTorch ModelZoo中属于您的数据集或模型，请您通过 Gitee 中提交 issue，您也可以联系ecosystem@cambricon.com告知我们。


