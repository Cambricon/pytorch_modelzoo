#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)
TIMM_DIR=$(cd ${CUR_DIR}/../models;pwd)

pushd $TIMM_DIR
if [ -z $IMAGENET_TRAIN_DATASET ]; then
    echo "[ERROR] Please set IMAGENET_TRAIN_DATASET."
    exit 1
fi
run_cmd="python -m torch.distributed.launch --nproc_per_node=8 train.py \
          $IMAGENET_TRAIN_DATASET \
          --model swin_tiny_patch4_window7_224 --batch-size 128 --epochs 300  \
          --opt adamw --opt-eps 1.0e-08 --opt-betas 0.9 0.999 --momentum 0.9  \
          --lr 0.001  --min-lr 1.0e-05 --warmup-epochs 20 --warmup-lr 1.0e-6 --weight-decay 0.05 \
          --decay-rate 0.1  --decay-epochs 30 --clip-grad 5.0 \
          --device mlu --amp --native-amp\
          --color-jitter 0.4 --cutmix 1.0 --mixup 0.8 --mixup-mode batch --mixup-prob 1.0 \
          --recount 1 --remode pixel --reprob 0.25 --train-interpolation bicubic  --seed 0\
          --drop-path 0.2 --pin-mem \
          --cooldown-epoch 0 --output ./SwinTransformer/tmp"
eval "${run_cmd}"
popd
