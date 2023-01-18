#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

export DATASET_NAME="ImageNet_2012"

if [ -z ${IMAGENET_TRAIN_DATASET} ]; then
        echo "please set environment variable IMAGENET_TRAIN_DATASET."
        exit 1
fi

pushd $CUR_DIR
export MLU_VISIBLE_DEVICES=0,1,2,3

bash test_benchmark.sh shufflenet_v2_x1_0-fp32-mlu-ci_train
bash test_benchmark.sh shufflenet_v2_x1_0-fp32-mlu-ci_eval

bash test_benchmark.sh shufflenet_v2_x1_0-fp32-mlu-ddp-ci_train
bash test_benchmark.sh shufflenet_v2_x1_0-fp32-mlu-ddp-ci_eval

popd
