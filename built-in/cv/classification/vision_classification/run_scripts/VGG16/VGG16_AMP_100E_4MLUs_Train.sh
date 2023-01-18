#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

if [ -z ${IMAGENET_TRAIN_DATASET} ]; then
  echo "please set environment variable IMAGENET_TRAIN_DATASET."
  exit 1
fi

pushd $CUR_DIR

export MLU_VISIBLE_DEVICES=0,1,2,3
python $CUR_DIR/../../classify_train.py \
	-a vgg16 \
	--iters -1 \
	--batch-size 128 \
	--lr 0.02 \
	--device mlu \
	--momentum 0.9 \
       	--wd 1e-4 \
       	--seed 42 \
	--data $IMAGENET_TRAIN_DATASET \
	--logdir $CUR_DIR/../../data/output/vgg16_amp_four_card_log \
	--epochs 100 \
	--save_ckp \
	--save_best \
	--ckpdir $CUR_DIR/../../data/output/vgg16_amp_four_card_ckps \
       	--multiprocessing-distributed \
	-j 8 \
	--dist-backend cncl \
	--world-size 1 \
	--rank 0 \
	--pyamp

popd