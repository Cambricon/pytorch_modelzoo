#!/bin/bash

set -e
echo "Setting envs......"
# 数据路径

export DATASET_NAME="KiTS19"

if [ -z ${PYTORCH_TRAIN_DATASET} ]; then
  echo "please set environment variable PYTORCH_TRAIN_DATASET."
  exit 1
fi

if [ -z ${PYTORCH_TRAIN_CHECKPOINT} ]; then
  echo "please set environment variable PYTORCH_TRAIN_CHECKPOINT."
  exit 1
fi

DATASET_ROOT=$PWD/data
export DATASET_DIR=$DATASET_ROOT/KiTS19

if [ ! -d ${DATASET_DIR} ]; then
  if [ ! -d $DATASET_ROOT ]; then
    mkdir -p $DATASET_ROOT
  fi
  ln -s ${PYTORCH_TRAIN_DATASET}/KiTS19/pre_data_dir ${DATASET_DIR}
fi
