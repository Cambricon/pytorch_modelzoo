#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
TRANS_DIR=$(cd ${CUR_DIR}/../models/;pwd)
if [ -z ${PYTORCH_TRAIN_DATASET} ]; then
  echo "please set environment variable PYTORCH_TRAIN_DATASET."
  exit 1
fi

pushd $TRANS_DIR
export MLU_VISIBLE_DEVICES=0,1,2,3
python train.py --adam --beta1 0.9 --lr 0.0001 --trainRoot ${PYTORCH_TRAIN_DATASET}/Synth90k/ --valRoot ${PYTORCH_TRAIN_DATASET}/Synth90k/ --expr_dir ${PYTORCH_TRAIN_CHECKPOINT}/crnn/checkpoints_fp/ --mlu --ddp True --ngpu 4 --batchSize 16 --workers 4 --nepoch 30 --iter -1 --displayInterval 1 --cudnn_lstm --saveInterval 5
popd
