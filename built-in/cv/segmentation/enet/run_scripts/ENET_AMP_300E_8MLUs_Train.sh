#!/bin/bash
CUR_DIR=$(cd $(dirname $0);pwd)
pushd ${CUR_DIR}/../models
if [ -z $PYTORCH_TRAIN_DATASET ]; then
  echo "[ERROR] Please set PYTORCH_TRAIN_DATASET."
  exit 1
fi

data=$PYTORCH_TRAIN_DATASET/CityScapes
save_dir=./checkpoints

MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
main.py -m train \
--deterministic \
--distributed   \
--eval  \
--pyamp \
--name enet \
--dataset cityscapes \
--dataset-dir $data    \
--save-dir $save_dir \
--batch-size 2    \
--learning-rate 0.0015  \
--device "mlu"   \
--dist-backend "cncl"   \
--lr-decay 0.5  \
--height 512    \
--width 1024    \
--seed 42

popd
