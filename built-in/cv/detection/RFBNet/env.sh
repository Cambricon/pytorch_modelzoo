#bin/bash
echo "Step1 Setting envs......"
export VOC_DATASET_PATH=datasets
export PROJ_DIR=$PWD
export VOC2007_TRAIN_DATASET=/data/datasets/VOC2007/
export VOC2012_TRAIN_DATASET=/data/datasets/VOCdevkit/VOC2012/
export PYTORCH_TRAIN_DATASET=/data/pytorch/datasets/
export PYTORCH_TRAIN_CHECKPOINT=/data/pytorch/models/pytorch_weight/checkpoints/

if [ -z ${VOC2007_TRAIN_DATASET} ]; then
  echo "please set environment variable VOC2007_TRAIN_DATASET."
  exit 1
fi

if [ -z ${VOC2012_TRAIN_DATASET} ]; then
  echo "please set environment variable VOC2012_TRAIN_DATASET."
  exit 1
fi

if [ -z ${PYTORCH_TRAIN_CHECKPOINT} ]; then
  echo "please set environment variable PYTORCH_TRAIN_CHECKPOINT."
  exit 1
fi

echo "VOC2007_TRAIN_DATASET is "$VOC2007_TRAIN_DATASET
echo "VOC2012_TRAIN_DATASET is "$VOC2012_TRAIN_DATASET
echo "PYTORCH_TRAIN_CHECKPOINT is "$PYTORCH_TRAIN_CHECKPOINT

cd $PROJ_DIR/models/
if [ ! -d  $VOC_DATASET_PATH/VOCdevkit ];then
    mkdir -p $VOC_DATASET_PATH/VOCdevkit
    ln -s $VOC2007_TRAIN_DATASET $VOC_DATASET_PATH/VOCdevkit/VOC2007
    ln -s $VOC2012_TRAIN_DATASET $VOC_DATASET_PATH/VOCdevkit/VOC2012
fi
cd ../