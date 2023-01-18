#bin/bash
set -e
echo "Setting envs......"
export PROJ_DIR=$PWD
export PYTORCH_TRAIN_DATASET=/data/pytorch/datasets/
export PYTORCH_TRAIN_CHECKPOINT=/data/pytorch/models/pytorch_weight/checkpoints/
export COCO2014_TRAIN_DATASET=${PYTORCH_TRAIN_DATASET}COCO2014

mkdir -p $PROJ_DIR/data/weights/gpu_checkpoints

if [ -z ${COCO2014_TRAIN_DATASET} ]; then
  echo "please set environment variable COCO2014_TRAIN_DATASET."
  exit 1
fi

if [ -z ${PYTORCH_TRAIN_CHECKPOINT} ]; then
  echo "please set environment variable PYTORCH_TRAIN_CHECKPOINT."
  exit 1
fi

echo "COCO2014_TRAIN_DATASET is "$COCO2014_TRAIN_DATASET
echo "PYTORCH_TRAIN_CHECKPOINT is "$PYTORCH_TRAIN_CHECKPOINT

cd $PROJ_DIR/models
if [ ! -d datasets ];then
    mkdir -p datasets
    ln -sf $COCO2014_TRAIN_DATASET datasets/coco
fi
cd ../

