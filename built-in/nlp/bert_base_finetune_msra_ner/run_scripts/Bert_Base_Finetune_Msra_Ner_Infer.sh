#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
BERT_DIR=$(cd ${CUR_DIR}/../models;pwd)

pushd $BERT_DIR

export MLU_VISIBLE_DEVICES=0

python evaluate.py --device mlu
popd
