#!/bin/bash
set -e
set -x
CUR_DIR=$(cd $(dirname $0);pwd)
TIMM_DIR=$(cd ${CUR_DIR}/../models;pwd)

pushd $TIMM_DIR
if [ -z $IMAGENET_TRAIN_DATASET ]; then
    echo "[ERROR] Please set IMAGENET_TRAIN_DATASET."
    exit 1
fi

if [ -z $TIMM_INFER_MODEL ]; then
    echo "[ERROR] Please set TIMM_INFER_MODEL."
    exit 1
fi

run_cmd="python validate.py \
          $IMAGENET_TRAIN_DATASET/val \
          --model swin_tiny_patch4_window7_224 \
	  --checkpoint $TIMM_INFER_MODEL \
          --device mlu"
eval "${run_cmd}"
popd
