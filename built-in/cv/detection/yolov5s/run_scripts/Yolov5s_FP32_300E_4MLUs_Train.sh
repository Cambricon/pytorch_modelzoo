if [ -z $MLU_VISIBLE_DEVICES ]; then
  export MLU_VISIBLE_DEVICES=0,1,2,3
fi

if [ -z $PYTORCH_TRAIN_DATASET ]; then
  echo "[ERROR] Please set PYTORCH_TRAIN_DATASET."
  exit 1
fi

CUR_DIR=$(cd $(dirname $0);pwd)

yolo_path=${CUR_DIR}/../models
epochs=300
data=data/coco.yaml
cfg=models/yolov5s.yaml

pushd $yolo_path
# create datasets dir soft link
if [ ! -d "../coco" ]; then
  ln -sf "$PYTORCH_TRAIN_DATASET/COCO2017" "../coco"
fi

# train
python ${yolo_path}/train.py --epochs ${epochs} --data ${data} --cfg ${cfg} --device "mlu" --multiprocessing-distributed --dist-backend cncl --notest

popd
