#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

if [ -z ${IMAGENET_TRAIN_DATASET} ]; then
  echo "please set environment variable IMAGENET_TRAIN_DATASET."
  exit 1
fi

if [ -z ${IMAGENET_INFER_CHECKPOINT} ]; then
  echo "please set environment variable IMAGENET_INFER_CHECKPOINT."
  exit 1
fi

pushd $CUR_DIR
python $CUR_DIR/../../classify_infer.py \
	--network shufflenet_v2_x1_5 \
	--batch_size 64 \
	--device mlu \
       	--input_data_type float32 \
	--data $IMAGENET_TRAIN_DATASET \
	--ckpt ${IMAGENET_INFER_CHECKPOINT} \
	-j 12
popd

