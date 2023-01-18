#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
OLTR_DIR=$(cd ${CUR_DIR}/../models;pwd)

export MASTER_ADDR='127.0.0.1'
export MASTER_PORT=29505

if [ -z ${PYTORCH_TRAIN_DATASET} ]; then
  echo "please set environment variable PYTORCH_TRAIN_DATASET."
  exit 1
fi

if [ -z ${IMAGENET_TRAIN_DATASET} ]; then
  echo "please set environment variable IMAGENET_TRAIN_DATASET."
  exit 1
fi

pushd $OLTR_DIR
# create datasets dir soft link
if [ ! -L "./data/ImageNet_LT" ]; then
  ln -sf "$PYTORCH_TRAIN_DATASET/ImageNet_LT" "./data/ImageNet_LT"
fi


echo "Training Stage1"
python main_imagenet.py --device 0 --data_path $IMAGENET_TRAIN_DATASET --config imagenet_mid_stage_1.py --seed 1
echo "Training Stage2"
python main_imagenet.py --device 0 --data_path $IMAGENET_TRAIN_DATASET --config imagenet_mid_stage_2_meta_embedding.py --seed 1

popd
