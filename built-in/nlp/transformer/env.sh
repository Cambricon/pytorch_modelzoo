# export MODEL_URL="https://www.dropbox.com/s/iqjiuw3hkdqa6td/model_epoch_18.pth?dl=0"

echo "Setting up envs..."

export DATASET_NAME="IWSLT2016"

if [ -z ${IWSLT_CORPUS_PATH} ]; then
  echo "please set environment variable IWSLT_CORPUS_PATH."
  exit 1
fi

if [ -z ${TRANSFORMER_CKPT} ]; then
  echo "please set environment variable TRANSFORMER_CKPT."
  exit 1
fi

echo "IWSLT_CORPUS_PATH is "$IWSLT_CORPUS_PATH
echo "TRANSFORMER_CKPT is "$TRANSFORMER_CKPT

export CORPORA_PATH=$PWD/models/corpora
export CKPT_MODEL_PATH=$PWD/models/ckpt_model

if [ ! -d ${CORPORA_PATH} ]; then
  ln -s ${IWSLT_CORPUS_PATH} ${CORPORA_PATH}
fi
if [ ! -d ${CKPT_MODEL_PATH} ]; then
  ln -s ${TRANSFORMER_CKPT} ${CKPT_MODEL_PATH}
fi
