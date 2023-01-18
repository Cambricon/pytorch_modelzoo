#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
TRANS_DIR=$(cd ${CUR_DIR}/../models/;pwd)

# set original weights files
weights=$PYTORCH_TRAIN_CHECKPOINT/yolov3/darknet53.conv.74
data=$TRANS_DIR/data/coco2014.data
cfg=$TRANS_DIR/cfg/yolov3.cfg
ckp_dir=$TRANS_DIR/weights/

pushd $TRANS_DIR
python test.py --img-size 416 --iou-thr 0.6 --task test --weights $ckp_dir/model_best.pth.tar --data $data --batch-size 32 --cfg $cfg --device "mlu"
popd
