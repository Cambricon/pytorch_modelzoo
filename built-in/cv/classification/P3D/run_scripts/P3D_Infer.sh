#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

if [ -z ${PYTORCH_TRAIN_DATASET} ]; then
  echo "please set environment variable PYTORCH_TRAIN_DATASET."
  exit 1
fi

if [ -z ${PYTORCH_INFER_CHECKPOINT} ]; then
  echo "please set environment variable PYTORCH_INFER_CHECKPOINT."
  exit 1
fi

pushd $CUR_DIR

python $CUR_DIR/../models/main.py ${PYTORCH_TRAIN_DATASET}/ucf101/ --evaluate ${PYTORCH_INFER_CHECKPOINT} --eval_steps -1 --logdir ${CUR_DIR}/../output/

popd
