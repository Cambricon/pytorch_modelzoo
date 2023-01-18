#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

export DATASET_NAME="LJSpeech-1.1"

pushd $CUR_DIR/../
source env.sh
popd

echo "start setup lib"
## 依赖库 需要sudo权限
pushd ${CUR_DIR}/../models/; bash lib.sh; popd;

pushd $CUR_DIR
export MLU_VISIBLE_DEVICES=0,1
bash test_benchmark.sh fp32-mlu-ddp-ci
bash test_benchmark.sh fp32-mlu-ci
popd
