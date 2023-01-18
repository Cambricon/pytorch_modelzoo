#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

export DATASET_NAME="facades"

pushd $CUR_DIR/../
source env.sh
popd

pushd $CUR_DIR
bash test_benchmark.sh fp32-mlu-ci
popd
