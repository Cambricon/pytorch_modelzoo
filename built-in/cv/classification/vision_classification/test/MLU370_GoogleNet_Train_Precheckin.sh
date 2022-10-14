#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

pushd $CUR_DIR
export MLU_VISIBLE_DEVICES=0,1,2,3

bash test_benchmark.sh googlenet-fp32-mlu-ci_train
bash test_benchmark.sh googlenet-fp32-mlu-ci_eval

bash test_benchmark.sh googlenet-fp32-mlu-ddp-ci_train
bash test_benchmark.sh googlenet-fp32-mlu-ddp-ci_eval

popd

