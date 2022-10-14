#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

pushd $CUR_DIR
export MLU_VISIBLE_DEVICES=0,1,2,3
bash $CUR_DIR/../../test/test_benchmark.sh shufflenet_v2_x1_0-fp32-mlu-ddp
popd

