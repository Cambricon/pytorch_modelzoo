echo "Setting up envs..."

if [ -z ${PYTORCH_TRAIN_DATASET} ]; then
  echo "please set environment variable PYTORCH_TRAIN_DATASET."
  exit 1
fi

echo "PYTORCH_TRAIN_DATASET is "$PYTORCH_TRAIN_DATASET
