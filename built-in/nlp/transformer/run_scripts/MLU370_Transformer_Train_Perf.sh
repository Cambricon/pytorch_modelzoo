#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

pushd $CUR_DIR/../test
export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash test_benchmark.sh fp32-mlu-ddp
popd
