#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

cd ../
source env.sh
pushd $CUR_DIR

export MLU_VISIBLE_DEVICES=0
export DATASET_NAME="VOC2007"
bash test_benchmark.sh fp32-mlu-ddp-ci
# bash test_benchmark.sh amp-mlu-ddp-ci
popd
