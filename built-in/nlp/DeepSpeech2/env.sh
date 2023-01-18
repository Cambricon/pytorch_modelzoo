#bin/bash
set -e
echo "Setting up envs..."

if [ -z ${PYTORCH_TRAIN_DATASET} ]; then
  echo "please set environment variable PYTORCH_TRAIN_DATASET."
  exit 1
fi

LibriSpeech_dataset_PATH=$PYTORCH_TRAIN_DATASET/LibriSpeech_dataset
LibriSpeech_dataset_LOCAL_PATH=$PWD/models/LibriSpeech_dataset
echo "LibriSpeech_dataset path is $LibriSpeech_dataset_PATH"

if [ ! -d ${LibriSpeech_dataset_LOCAL_PATH} ]; then
  ln -s ${LibriSpeech_dataset_PATH} ${LibriSpeech_dataset_LOCAL_PATH}
fi
