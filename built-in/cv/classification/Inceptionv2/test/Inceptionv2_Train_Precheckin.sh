#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

export DATASET_NAME="ImageNet_2012"

pushd $CUR_DIR/../models
popd

pushd $CUR_DIR
export MLU_VISIBLE_DEVICES=0,1
bash test_benchmark.sh fp32-mlu-ddp-ci
bash test_benchmark.sh fp32-mlu-ci
popd

