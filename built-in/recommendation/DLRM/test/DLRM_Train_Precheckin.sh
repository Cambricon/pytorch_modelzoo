#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
export DATASET_NAME="MovieLens"

pushd $CUR_DIR/../
source env.sh
popd

echo "BENCHMARK_LOG is "$BENCHMARK_LOG
echo "AVG_LOG is "$AVG_LOG

pushd $CUR_DIR
export MLU_VISIBLE_DEVICES=0,1
bash test_benchmark.sh fp32-mlu-ddp-ci_train
bash test_benchmark.sh fp32-mlu-ci_train
popd
