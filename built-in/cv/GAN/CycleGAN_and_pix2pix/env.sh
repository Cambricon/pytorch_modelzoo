echo "Setting up envs..."

if [ -z ${PYTORCH_TRAIN_DATASET} ]; then
  echo "please set environment variable PYTORCH_TRAIN_DATASET."
  exit 1
fi

export PIX2PIX_DATASET_PATH=$PYTORCH_TRAIN_DATASET/facades

echo "pix2pix dataset path is $PIX2PIX_DATASET_PATH"

export FACADES_PATH=$PWD/models/facades

if [ ! -d ${FACADES_PATH} ]; then
  ln -s ${PIX2PIX_DATASET_PATH} ${FACADES_PATH}
fi
