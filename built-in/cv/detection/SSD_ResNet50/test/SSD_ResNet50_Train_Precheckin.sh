#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

cd ../
source env.sh
pushd $CUR_DIR

export DATASET_NAME="COCO2017"
export MLU_VISIBLE_DEVICES=0,1,2,3

bash test_benchmark.sh fp32-mlu-ddp
popd
