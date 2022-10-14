# PyTorch ModelZoo 

## 介绍

PyTorch是时下最流行的深度学习框架，寒武纪对其进行了定制化开发，新增了对寒武纪加速板卡及寒武纪深度学习软件栈的支持，通常称之为Cambricon PyTorch。相比于原生PyTorch，用户基本不用做任何代码改动即可快速地将深度学习模型迁移至Cambricon PyTorch上。

针对CV 分类、检测、分割、NLP、语音等场景常用的各类经典和前沿的深度学习模型，本仓库展示了如何对其进行适配，使其可运行在Cambricon PyTorch上。开发者在进行其他AI 应用移植时可参考本仓库。


## 网络支持列表和链接

CV：

| MODELS | FRAMEWORK | Train Mode |Distributed Train| Infer  Mode
| ------------- | ------------- | ------------- | ------------- | ------------- | 
| [ResNet50](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32&&AMP|YES| Torch2MM/CNNL |
| [ResNet18](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32|YES| Torch2MM/CNNL | 
| [VGG16](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32&&AMP|YES| Torch2MM/CNNL | 
| [MobileNetv2](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32&&AMP|YES| Torch2MM/CNNL | 
| [AlexNet](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32|YES| Torch2MM/CNNL | 
| [GoogLeNet](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32&&AMP|YES| Torch2MM/CNNL | 
| [ResNet101](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32|YES| Torch2MM/CNNL | 
| [VGG19](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32|YES| Torch2MM/CNNL | 
| [VGG16_bn](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32|YES| Torch2MM/CNNL |
| [ShuffleNet_v2_x0_5](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32|YES| Torch2MM/CNNL |
| [ShuffleNet_v2_x1_0](built-in/cv/classification/vision_classification) | PyTorch1.6|FP32|YES| Torch2MM/CNNL |


NLP:

| MODELS | FRAMEWORK | Train Mode |Distributed Train| Infer  Mode
| ------------- | ------------- | ------------- | ------------- | ------------- | 
| [transformer](built-in/nlp/transformer) | PyTorch1.6|FP32|YES| CNNL |



## issues/wiki/forum 跳转链接

## contrib 指引和链接

## LICENSE

PyTorch ModelZoo  的 License 具体内容请参见[LICENSE](LICENSE)文件。

## 免责声明

PyTorch ModelZoo 仅提供公共数据集以及预训练模型的下载链接，公共数据集及预训练模型并不属于 PyTorch ModelZoo ，PyTorch ModelZoo  也不对其质量或维护承担责任。请您在使用公共数据集和预训练模型的过程中，确保符合其对应的使用许可。

如果您不希望您的数据集或模型公布在 PyTorch ModelZoo上，或者您希望更新 PyTorch ModelZoo中属于您的数据集或模型，请您通过 Gitee 中提交 issue，您也可以联系ecosystem@cambricon.com告知我们。

