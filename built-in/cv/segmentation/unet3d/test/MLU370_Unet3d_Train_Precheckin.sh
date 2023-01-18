#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

pushd $CUR_DIR/../
source env.sh
popd

pushd $CUR_DIR
export MLU_VISIBLE_DEVICES=0,1,2,3
bash test_benchmark.sh fp32-mlu-ddp-ci
popd
