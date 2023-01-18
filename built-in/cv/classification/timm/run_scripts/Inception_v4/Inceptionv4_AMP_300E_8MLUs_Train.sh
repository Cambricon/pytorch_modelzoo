#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)
TIMM_DIR=$(cd ${CUR_DIR}/../../models;pwd)

export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

if [ -z ${IMAGENET_TRAIN_DATASET} ]; then
    echo "[ERROR] Please set environment variable IMAGENET_TRAIN_DATASET."
    exit 1
fi

if [ -z ${PTH_AND_LOG_DIR} ]; then
  echo "[ERROR] Please set environment variable PTH_AND_LOG_DIR."
  exit 1
fi

pushd $TIMM_DIR

DATA_DIR=$IMAGENET_TRAIN_DATASET
OUTPUT_DIR=$PTH_AND_LOG_DIR

export WORLD_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=12346

python -m torch.distributed.launch --nproc_per_node=8 --master_port $MASTER_PORT   \
          train.py                    \
          $DATA_DIR                   \
          --model inception_v4        \
          -b 200                      \
          --sched step                \
          --epochs 300                \
          --decay-epochs 2.4          \
          --decay-rate .973           \
          --opt rmsproptf             \
          --opt-eps .001              \
          -j 7                        \
          --warmup-lr 3.5e-7          \
          --weight-decay 1e-5         \
          --drop 0.2                  \
          --model-ema                 \
          --model-ema-decay 0.9999    \
          --aa rand-m9-mstd0.5        \
          --remode pixel              \
          --reprob 0.2                \
          --lr 0.0224                 \
          --lr-noise 0.42 0.9         \
          --device mlu                \
          --amp                       \
          --output $OUTPUT_DIR        

popd
