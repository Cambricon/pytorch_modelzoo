#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

SEED=0

MAX_EPOCHS=4000
QUALITY_THRESHOLD="0.908"
START_EVAL_AT=1000
EVALUATE_EVERY=20
LEARNING_RATE="3.2"
LR_WARMUP_EPOCHS=200
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1

if [ -d ${DATASET_DIR} ] && [ -d ${PYTORCH_TRAIN_CHECKPOINT} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"

# CLEAR YOUR CACHE HERE
pushd ${CUR_DIR}/../models
  python -c "
from mlperf_logging.mllog import constants
from runtime.logging import mllog_event
mllog_event(key=constants.CACHE_CLEAR, value=True)"


export WORLD_SIZE=4


  python -m torch.distributed.launch --nproc_per_node ${WORLD_SIZE} main.py \
    --data_dir ${DATASET_DIR} \
    --epochs ${MAX_EPOCHS} \
    --evaluate_every ${EVALUATE_EVERY} \
    --start_eval_at ${START_EVAL_AT} \
    --quality_threshold ${QUALITY_THRESHOLD} \
    --batch_size ${BATCH_SIZE} \
    --optimizer sgd \
    --ga_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --seed ${SEED} \
    --lr_warmup_epochs ${LR_WARMUP_EPOCHS} \
    --save_ckpt_path  ${PYTORCH_TRAIN_CHECKPOINT}/unet3d_4000.pth 

        # end timing
        end=$(date +%s)
        end_fmt=$(date +%Y-%m-%d\ %r)
        echo "ENDING TIMING RUN AT $end_fmt"


        # report result
        result=$(( $end - $start ))
        result_name="image_segmentation"


        echo "RESULT,$result_name,$SEED,$result,$USER,$start_fmt"
else
        echo "Directory ${DATASET_DIR} or ${PYTORCH_TRAIN_CHECKPOINT} does not exist, please check it"
fi

popd
