#bin/bash
export PROJ_DIR=$PWD
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_TRAIN_DATASET=/data/pytorch/datasets/
export PYTORCH_TRAIN_CHECKPOINT=/data/pytorch/models/pytorch_weight/checkpoints/

mkdir -p $PROJ_DIR/data/weights/gpu_checkpoints

if [ -z ${PYTORCH_TRAIN_DATASET} ]; then
  echo "please set environment variable PYTORCH_TRAIN_DATASET."
  exit 1
fi

if [ -z ${PYTORCH_TRAIN_CHECKPOINT} ]; then
  echo "please set environment variable PYTORCH_TRAIN_CHECKPOINT."
  exit 1
fi

echo "PYTORCH_TRAIN_DATASET is "$PYTORCH_TRAIN_DATASET
echo "PYTORCH_TRAIN_CHECKPOINT is "$PYTORCH_TRAIN_CHECKPOINT

