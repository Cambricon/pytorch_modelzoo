#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
TRANS_DIR=$(cd ${CUR_DIR}/../models/;pwd)

pushd $TRANS_DIR
export MLU_VISIBLE_DEVICES=0
python train.py --ckp-path ${CKPT_MODEL_PATH} --num_epochs 20 --opt_level O0
popd
