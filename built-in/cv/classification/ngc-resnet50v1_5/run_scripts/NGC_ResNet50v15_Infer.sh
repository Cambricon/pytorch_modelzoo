#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
RES_DIR=$(cd ${CUR_DIR}/../models;pwd)

if [ -z ${IMAGENET_TRAIN_DATASET} ]; then
  echo "please set environment variable IMAGENET_TRAIN_DATASET."
  exit 1
fi

pushd $RES_DIR

python main.py $IMAGENET_TRAIN_DATASET --raport-file raport.json -j1 -p 100 --arch resnet50 -c fanin --workspace ${1:-./} -b 128 --epochs 91 --resume checkpoint.pth.tar --evaluate

popd
