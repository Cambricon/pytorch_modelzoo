#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
export DATASET_NAME="LJSpeech-1.1"
pushd $CUR_DIR/../
source env.sh
popd

pushd $CUR_DIR
export MLU_VISIBLE_DEVICES=0,1,2,3

bash test_benchmark.sh WaveGlow-fp32-mlu-ddp-ci
bash test_benchmark.sh WaveGlow-fp32-mlu-ci


popd
