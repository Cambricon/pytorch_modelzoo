#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

if [ -z ${IMAGENET_TRAIN_DATASET} ]; then
  echo "please set environment variable IMAGENET_TRAIN_DATASET."
  exit 1
fi

pushd $CUR_DIR
export MLU_VISIBLE_DEVICES=0,1,2,3

bash test_benchmark.sh inception_v3-fp32-mlu-ddp-ci
bash test_benchmark.sh inception_v3-amp-mlu-ddp-ci
bash test_benchmark.sh inception_v3-fp32-mlu-ci

popd

