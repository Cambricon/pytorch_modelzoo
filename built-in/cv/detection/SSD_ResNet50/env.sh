#bin/bash
echo "Setting up envs..."
export PROJ_DIR=$PWD
export PYTORCH_TRAIN_DATASET=/data/pytorch/datasets/
export PYTORCH_TRAIN_CHECKPOINT=/data/pytorch/models/pytorch_weight/checkpoints/
export COCO2017_TRAIN_DATASET=$PYTORCH_TRAIN_DATASET/COCO2017

if [ -z ${COCO2017_TRAIN_DATASET} ]; then
  echo "please set environment variable COCO2017_TRAIN_DATASET."
  exit 1
fi

if [ -z ${PYTORCH_TRAIN_CHECKPOINT} ]; then
  echo "please set environment variable PYTORCH_TRAIN_CHECKPOINT."
  exit 1
fi

echo "COCO2017_TRAIN_DATASET is "$COCO2017_TRAIN_DATASET
echo "PYTORCH_TRAIN_CHECKPOINT is "$PYTORCH_TRAIN_CHECKPOINT
