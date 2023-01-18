#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
WORK_DIR=$(cd ${CUR_DIR}/../../models/;pwd)

pushd $WORK_DIR

python inference.py  -i phrases/phrase.txt -o output/ --tacotron2 output/checkpoint_Tacotron2_last.pt  --device mlu

popd
