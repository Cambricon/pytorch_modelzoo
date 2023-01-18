#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

export DATASET_NAME="COCO2017"

if [ -z ${PYTORCH_TRAIN_CHECKPOINT} ]; then 
    export PYTORCH_TRAIN_CHECKPOINT=/data/pytorch/models/pytorch_weight/checkpoints
fi

pushd $CUR_DIR
export MLU_VISIBLE_DEVICES=0,1,2,3

bash test_benchmark.sh fp32-mlu-ddp-ci_train
bash test_benchmark.sh fp32-mlu-ci_train

popd

