#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)
TIMM_DIR=$(cd ${CUR_DIR}/../../models;pwd)

if [ -z ${IMAGENET_TRAIN_DATASET} ]; then
    echo "[ERROR] Please set environment variable IMAGENET_TRAIN_DATASET."
    exit 1
fi

if [ -z ${PTH_AND_LOG_DIR} ]; then
  echo "[ERROR] Please set environment variable PTH_AND_LOG_DIR."
  exit 1
fi

pushd $TIMM_DIR

export MLU_VISIBLE_DEVICES=0

DATA_DIR=$IMAGENET_TRAIN_DATASET
OUTPUT_DIR=$PTH_AND_LOG_DIR

## 需要从OUTPUT_DIR下选择一个保存的权重
EVAL_CKPT=path/to/xxx.pth.tar

python validate.py               \
       $DATA_DIR                 \
       --model inception_v4      \
       -b 512                    \
       -j 7                      \
       --device mlu              \
       --checkpoint $EVAL_CKPT                     

popd
