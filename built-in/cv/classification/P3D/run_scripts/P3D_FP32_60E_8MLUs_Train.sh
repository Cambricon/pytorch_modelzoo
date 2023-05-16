#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)
P3D_DIR=$(cd ${CUR_DIR}/../models;pwd)

if [ -z ${PYTORCH_TRAIN_DATASET} ]; then
  echo "please set environment variable PYTORCH_TRAIN_DATASET."
  exit 1
fi

if [ -z ${PYTORCH_TRAIN_CHECKPOINT} ]; then
  echo "please set environment variable PYTORCH_TRAIN_CHECKPOINT."
  exit 1
fi


pushd $P3D_DIR
num_card=8
export WORLD_SIZE=${num_card}
export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --master_port 29502 --nproc_per_node=8 main.py ${PYTORCH_TRAIN_DATASET}/ucf101/ --num-dev=${num_card} --batch-size 16 --lr 1e-3 --device_param mlu --dropout 0.9 --seed 42 --print-freq=50 --dist-backend cncl --pretrained --early-stop=1000 --epoch=60
popd
