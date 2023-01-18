#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
TRANS_DIR=$(cd ${CUR_DIR}/../models/;pwd)

# set original weights files
weights=$PYTORCH_TRAIN_CHECKPOINT/yolov3/darknet53.conv.74
data=$TRANS_DIR/data/coco2014.data
cfg=$TRANS_DIR/cfg/yolov3.cfg
ckp_dir=$TRANS_DIR/weights/
log_dir=$TRANS_DIR/logs/
batch_size=64

pushd $TRANS_DIR
export MLU_VISIBLE_DEVICES=0,1,2,3
python train.py --distributed --dist-backend cncl --batch-size $batch_size --weights $weights --cfg $cfg --data $data --ckp-path $ckp_dir --logdir $log_dir --notest --device "mlu" --pyamp
popd