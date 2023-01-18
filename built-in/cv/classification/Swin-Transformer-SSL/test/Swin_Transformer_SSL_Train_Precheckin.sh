#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

if [ -z ${IMAGENET_TRAIN_DATASET} ]; then
  echo "please set environment variable IMAGENET_TRAIN_DATASET."
  exit 1
fi

## 该环境变量的使用详见params_config.sh
if [ -z ${IMAGENET_TRAIN_CHECKPOINT} ]; then
  echo "please set environment variable IMAGENET_TRAIN_CHECKPOINT."
  exit 1
fi

pushd $CUR_DIR
export MLU_VISIBLE_DEVICES=0,1,2,3

bash test_benchmark.sh fp32-mlu-ddp-ci
bash test_benchmark.sh amp-mlu-ddp-ci
bash test_benchmark.sh fp32-mlu-ci

popd

