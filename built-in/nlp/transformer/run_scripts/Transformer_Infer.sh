#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
TRANS_DIR=$(cd ${CUR_DIR}/../models/;pwd)

pushd $TRANS_DIR
export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python eval.py --pretrained ${CKPT_MODEL_PATH}/model_epoch_09.pth
popd
