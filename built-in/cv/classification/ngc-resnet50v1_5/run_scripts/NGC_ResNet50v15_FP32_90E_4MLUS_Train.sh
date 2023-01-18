#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
RES_DIR=$(cd ${CUR_DIR}/../models;pwd)

if [ -z ${IMAGENET_TRAIN_DATASET} ]; then
  echo "please set environment variable IMAGENET_TRAIN_DATASET."
  exit 1
fi

pushd $RES_DIR

export MLU_VISIBLE_DEVICES=0,1,2,3
python ./multiproc.py --nproc_per_node 4 ./main.py $IMAGENET_TRAIN_DATASET --raport-file raport.json -j4 -p 100 --lr 2.048 --optimizer-batch-size 2048 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace ${1:-./} -b 128 --epochs 90 --training-only --dist-backend cncl --device mlu
popd
