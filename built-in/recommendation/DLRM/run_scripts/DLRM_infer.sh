#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

BASEDIR=${PYTORCH_TRAIN_DATASET}
DATASET=${DATASET:-ml-20m}
USER_MUL=${USER_MUL:-4}
ITEM_MUL=${ITEM_MUL:-16}
if [ -z ${PYTORCH_INFER_CHECKPOINT} ]; then
    echo "please set environment variable PYTORCH_INFER_CHECKPOINT."
    exit 1
fi

DATASET_DIR=${BASEDIR}/${DATASET}x${USER_MUL}x${ITEM_MUL}

pushd $CUR_DIR/../models/recommendation/pytorch
pip install -r requirements.txt
pip install "git+https://github.com/mlperf/logging.git"

if [ -d ${DATASET_DIR} ]
then
	python  ncf.py \
		--data ${DATASET_DIR} \
		--resume ${PYTORCH_INFER_CHECKPOINT} \
		-l 0.0002      \
		-b 65536      \
		--layers 256 256 128 64  \
		-f 64     \
		--seed 0  \
		--save_ckp 1 \
		--threshold 1.0 \
		--user_scaling ${USER_MUL}  \
		--item_scaling ${ITEM_MUL} \
		--cpu_dataloader   \
		--random_negatives  \
		--device mlu \
		--workers 8 \
		--do_predict \
		--multiprocessing-distributed
else
	echo "Directory ${DATASET_DIR} does not exist"
fi

popd
