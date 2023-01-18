# Pix2Pix的MLU训练

  本项目关于PyTorch的Pix2Pix网络的MLU训练。

  原生仓库: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/003efc4c8819de47ff11b5a0af7ba09aee7f5fc1。

  原生运行方法，参考 README.md。


## 环境准备:

- Cambricon PyTorch 1.9.0
- MLU 370

## from scratch训练 :
```
bash ./cambricon/from_srcatch_1mlu.sh
```

## 修改说明：

### 参数修改

新增参数说明：

1."--device"， 当前运行的设备环境，可选MLU、GPU、CPU， 默认

2."--resume_dir"，resume训练时加载checkpoint的文件路径，默认为空时，用保存checkpoint文件的路径。

3."--iter"，训练时运行多少次迭代，输入<=0的值的话默认是关闭，即运行完整个epoch

3."--seed"，设置随机种子，默认不设置。


### 功能修改

1. GPU默认不开启TF32模式，torch.backends.cuda.matmul.allow_tf32=False, torch.backends.cudnn.allow_tf32 = False

2. GPU设置随机种子后，会默认排除卷积算法随机性，torch.backends.cudnn.deterministic = True，torch.backends.cudnn.benchmark = False


## 特殊说明

1. MLU不支持运行DP模式。
