#!/bin/bash
set -e

if [ -z $PYTORCH_TRAIN_DATASET ]; then
  echo "[ERROR] Please set PYTORCH_TRAIN_DATASET."
  exit 1
fi

CUR_DIR=$(cd $(dirname $0);pwd)

yolo_path=${CUR_DIR}/../models
data=data/coco.yaml
cfg=models/yolov5s.yaml

pushd $yolo_path
# create datasets dir soft link
if [ ! -d "../coco" ]; then
  ln -sf "$PYTORCH_TRAIN_DATASET/COCO2017" "../coco"
fi

# test
weights=${yolo_path}/weights/mlu/checkpoint.pth.tar
python ${yolo_path}/test.py --data ${data} --weights ${weights} --device "mlu"

popd
