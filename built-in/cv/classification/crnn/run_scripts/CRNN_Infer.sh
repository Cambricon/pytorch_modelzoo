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
python test.py --valRoot $PYTORCH_TRAIN_DATASET/Synth90k/ --mlu --pretrained $PYTORCH_TRAIN_CHECKPOINT/crnn/checkpoints_fp/netCRNN_MLU_Final.pth --workers 8 --n_test_disp 40 --batchSize 512 --cudnn_lstm 
popd
