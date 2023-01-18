#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>
CUR_DIR=$(cd $(dirname $0);pwd)
SEED=0

MAX_EPOCHS=4000
QUALITY_THRESHOLD="0.903"
START_EVAL_AT=1400
EVALUATE_EVERY=10
LEARNING_RATE="3.2"
LR_WARMUP_EPOCHS=200
LR_DECAY_EPOCHS="1500 3000"
LR_DECAY_FACTOR="0.5"
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


export WORLD_SIZE=8


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
    --lr_decay_epochs ${LR_DECAY_EPOCHS} \
    --lr_decay_factor ${LR_DECAY_FACTOR}  \
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
        echo "Directory ${PYTORCH_TRAIN_CHECKPOINT} or ${DATASET_DIR} or XXX.pth does not exist, please check it"
fi


popd
