#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)
TIMM_DIR=$(cd ${CUR_DIR}/../../models;pwd)

export MLU_VISIBLE_DEVICES=0,1,2,3

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

export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=32136

python -m torch.distributed.launch --nproc_per_node=4  --master_port $MASTER_PORT   \
          train.py                    \
          $DATA_DIR                   \
          --model inception_v3        \
          -b 64                      \
          --sched cosine              \
          --epochs 200                \
          --decay-epochs 4           \
          --decay-rate 0.94           \
          --opt sgd                   \
          --opt-eps .001              \
          -j 8                        \
          --warmup-lr 0.0001          \
          --weight-decay 4e-5         \
          --drop 0.0                  \
          --aa rand-m9-mstd0.5        \
          --remode const              \
          --reprob 0.                \
          --lr 0.045                 \
          --device mlu                \
          --output $OUTPUT_DIR        

popd

