#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)
WAVERNN_DIR=$(cd ${CUR_DIR}/../models;pwd)

pushd $WAVERNN_DIR

if [ ! -d "./dataset/LJSpeech-1.1" ]; then
    echo "[ERROR] Please set LJSpeech-1.1."
    exit 1
fi

if [ ! -d "./dataset/data" ]; then
    python preprocess.py
fi

export MLU_VISIBLE_DEVICES=0

echo "Testing on MLU"
python val.py --device MLU --batch-size 32  \
       --checkpoint-path ./output/checkpoint_WaveRNN_10.pt 
popd
