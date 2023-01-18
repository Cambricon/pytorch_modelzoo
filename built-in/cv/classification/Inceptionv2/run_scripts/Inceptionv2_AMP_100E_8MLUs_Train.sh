#/bin/bash
if [ -z ${IMAGENET_TRAIN_DATASET} ]; then
  echo "please set environment variable IMAGENET_TRAIN_DATASET."
  exit 1
fi

export MLU_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
CUR_DIR=$(cd $(dirname $0);pwd)
pushd ${CUR_DIR}/../models/
python src/train.py \
    --config configs/inception_v2/inception_v2_train_ddp.yaml \
    --distributed \
    --train-dataset $IMAGENET_TRAIN_DATASET/train \
    --valid-dataset $IMAGENET_TRAIN_DATASET/val \
    --device mlu \
    --dist-backend cncl \
    --pyamp
popd
