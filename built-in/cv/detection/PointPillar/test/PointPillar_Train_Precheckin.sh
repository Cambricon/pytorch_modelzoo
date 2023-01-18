#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

export DATASET_NAME="nuscenes"
if [ -z ${PYTORCH_TRAIN_CHECKPOINT} ]; then
  echo "please set environment variable PYTORCH_TRAIN_CHECKPOINT."
  exit 1
fi
echo "PYTORCH_TRAIN_CHECKPOINT is "$PYTORCH_TRAIN_CHECKPOINT

pushd $CUR_DIR/../
source env.sh
popd

pushd $CUR_DIR
export MLU_VISIBLE_DEVICES=0,1,2,3
bash test_benchmark.sh fp32-mlu-ddp
bash test_benchmark.sh fp32-mlu-ddp-ci_train
popd
