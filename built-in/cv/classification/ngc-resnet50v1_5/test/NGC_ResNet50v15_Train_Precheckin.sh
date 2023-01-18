#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

export DATASET_NAME="ImageNet-2012"

if [ -z ${PYTORCH_TRAIN_DATASET} ]; then
  echo "please set environment variable PYTORCH_TRAIN_DATASET."
  exit 1
fi

if [ -z ${IMAGENET_TRAIN_CHECKPOINT} ]; then 
  export IMAGENET_TRAIN_CHECKPOINT=/data/pytorch/models/pytorch_weight/checkpoints/
fi

pushd $CUR_DIR
export MLU_VISIBLE_DEVICES=0,1,2,3

bash test_benchmark.sh fp32-mlu-ddp-ci
bash test_benchmark.sh fp32-mlu-ci

popd

