#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

cd ../
source env.sh
cd $PROJ_DIR/models
python setup.py build develop
pushd $CUR_DIR
export DATASET_NAME="COCO2014"
export MLU_VISIBLE_DEVICES=0,1,2,3

bash test_benchmark.sh fp32-mlu-ddp-ci
popd
