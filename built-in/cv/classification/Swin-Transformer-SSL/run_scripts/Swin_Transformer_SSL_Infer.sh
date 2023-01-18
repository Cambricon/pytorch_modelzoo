#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
SWIN_DIR=$(cd ${CUR_DIR}/../models;pwd)

if [ -z ${IMAGENET_TRAIN_DATASET} ]; then
  echo "please set environment variable IMAGENET_TRAIN_DATASET."
  exit 1
fi

if [ -z ${PTH_AND_LOG_DIR} ]; then
  echo "please set environment variable PTH_AND_LOG_DIR."
  exit 1
fi

pushd $SWIN_DIR

export MLU_VISIBLE_DEVICES=0

CONFIG=./configs/moby_swin_tiny.yaml
DATA_DIR=$IMAGENET_TRAIN_DATASET
OUTPUT_DIR=$PTH_AND_LOG_DIR

## 需要从OUTPUT_DIR下选择一个第一阶段训练保存的权重
PRETRAINED_CKPT=path/to/xxx.pth

## 需要从OUTPUT_DIR下选择一个第二阶段训练保存的权重
RESUME_LINEAR=path/to/yyy.pth

BATCH_SIZE=64

python moby_linear.py \
    --cfg $CONFIG \
    --device="mlu" \
    --data-path $DATA_DIR \
    --batch-size $BATCH_SIZE \
    --output $OUTPUT_DIR \
    --pretrained-ckpt $PRETRAINED_CKPT \
    --resume $RESUME_LINEAR \
    --eval

popd

