CURRENT_DIR=$(dirname $(readlink -f $0))
EXAMPLES_DIR=$(dirname $(dirname $(readlink -f $0)))

CORPORA_PATH=${EXAMPLES_DIR}/corpora
CKPT_MODEL_PATH=${EXAMPLES_DIR}/ckpt_model

# env
if [ -z ${IWSLT_CORPUS_PATH} ]; then
  echo "please set environment variable IWSLT_CORPUS_PATH."
  exit 1
fi
if [ -z ${TRANSFORMER_CKPT} ]; then
  echo "please set environment variable TRANSFORMER_CKPT."
  exit 1
fi

if [ ! -d ${CORPORA_PATH} ]; then
  ln -s ${IWSLT_CORPUS_PATH} ${CORPORA_PATH}
fi
if [ ! -d ${CKPT_MODEL_PATH} ]; then
  ln -s ${TRANSFORMER_CKPT} ${CKPT_MODEL_PATH}
fi

# param
device='MLU'
iterations=10
num_epochs=10
resume="${CKPT_MODEL_PATH}/model_epoch_09.pth"
ckpt_dir=${EXAMPLES_DIR}/ckpt_model
log_dir=${EXAMPLES_DIR}/logs

rm -rf ${log_dir} &>/dev/null

# run
pushd $EXAMPLES_DIR
python train.py  \
  --log-path ${log_dir} \
  --resume ${resume} \
  --num_epochs ${num_epochs} \
  --device ${device} \
  --iterations ${iterations} \
  --bitwidth 16 \
  --print-freq 1 \
  --dropout_rate 0.0
popd

# R2
pushd $EXAMPLES_DIR
python scripts/compute_R2.py ${CKPT_MODEL_PATH}/logs/ ${log_dir} ${num_epochs}
popd
