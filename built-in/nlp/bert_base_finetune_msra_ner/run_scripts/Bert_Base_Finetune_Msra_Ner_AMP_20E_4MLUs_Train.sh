#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
BERT_DIR=$(cd ${CUR_DIR}/../models;pwd)

if [ -z ${PYTORCH_TRAIN_CHECKPOINT} ]; then
  echo "please set environment variable PYTORCH_TRAIN_CHECKPOINT."
  exit 1
fi

pushd $BERT_DIR

export MLU_VISIBLE_DEVICES=0,1,2,3

BERT_PRETRAIN=$PYTORCH_TRAIN_CHECKPOINT

python -m torch.distributed.launch --nproc_per_node=4 --master_port=29000  \
       train_ddp.py \
       --distributed  \
       --nproc_per_node 4  \
       --bert_model_dir $BERT_PRETRAIN/bert-base-chinese-pytorch \
       --dist-backend cncl  \
       --device mlu  \
       --pyamp
popd
