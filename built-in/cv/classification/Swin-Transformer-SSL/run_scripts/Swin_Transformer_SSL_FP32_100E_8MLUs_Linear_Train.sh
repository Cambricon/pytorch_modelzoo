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

export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CONFIG=./configs/moby_swin_tiny.yaml
DATA_DIR=$IMAGENET_TRAIN_DATASET
OUTPUT_DIR=$PTH_AND_LOG_DIR

## 需要从OUTPUT_DIR下选择一个第一阶段训练保存的权重
PRETRAINED_CKPT=path/to/xxx.pth
BATCH_SIZE=64

export WORLD_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=12345

python -m torch.distributed.launch --nproc_per_node 8 --master_port $MASTER_PORT \
    moby_linear.py \
    --cfg $CONFIG \
    --device="mlu" \
    --distributed \
    --data-path $DATA_DIR \
    --batch-size $BATCH_SIZE \
    --output $OUTPUT_DIR \
    --pretrained-ckpt $PRETRAINED_CKPT 

popd

