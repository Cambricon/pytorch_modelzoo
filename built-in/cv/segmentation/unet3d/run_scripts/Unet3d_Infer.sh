#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

SEED=0
EXEC_MODE="evaluate"

if [ -d ${PYTORCH_TRAIN_CHECKPOINT} ] && [ -d ${DATASET_DIR} ]
then
    # start timing
      start=$(date +%s)
      start_fmt=$(date +%Y-%m-%d\ %r)
      echo "STARTING TIMING RUN AT $start_fmt"

pushd ${CUR_DIR}/../models

export WORLD_SIZE=1

  python -m torch.distributed.launch --nproc_per_node ${WORLD_SIZE} main.py \
    --data_dir ${DATASET_DIR} \
    --exec_mode ${EXEC_MODE} \
    --load_ckpt_path ${PYTORCH_TRAIN_CHECKPOINT}unet3d/unet3d_4000.pth
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
