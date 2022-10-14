#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

if [ -z ${IMAGENET_TRAIN_DATASET} ]; then
  echo "please set environment variable IMAGENET_TRAIN_DATASET."
  exit 1
fi

pushd $CUR_DIR

python $CUR_DIR/../../classify_infer.py --network vgg16_bn --batch_size 64 --device mlu --fusion_backend torch2mm  --input_data_type float32 --data $IMAGENET_TRAIN_DATASET -j 12 --iters 300 --do_benchmark true

popd
