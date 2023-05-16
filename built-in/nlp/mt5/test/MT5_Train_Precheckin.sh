#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

echo "BENCHMARK_LOG is "$BENCHMARK_LOG
echo "AVG_LOG is "$AVG_LOG

pushd $CUR_DIR
export MLU_VISIBLE_DEVICES=0,1
bash test_benchmark.sh fp32-mlu-ddp-ci
bash test_benchmark.sh fp32-mlu-ci
popd