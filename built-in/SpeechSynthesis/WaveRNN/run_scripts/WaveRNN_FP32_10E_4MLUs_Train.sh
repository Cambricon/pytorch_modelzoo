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

export MLU_VISIBLE_DEVICES=0,1,2,3

echo "Training-FP32 on MLU"
python -m torch.distributed.launch --nproc_per_node=4 \
       train.py  \
       --device MLU  \
       --do-train  \
       --seed 123456  \
       --epochs 10   \
       --iterations 1000  \
       --lr 1e-4   \
       --batch-size 32  \
       --dist-url 23456   \
       --num-per-checkpoint 1
popd
