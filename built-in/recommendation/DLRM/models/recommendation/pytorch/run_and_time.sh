#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

function usage
{
    echo "Usage:"
    echo "-------------------------------------------------------------"
    echo "|  $0 [seed] [device]"
    echo "|  parameter1: seed number, Seed 0 has been shown to converge deterministically."
    echo "|  parameter2: 0)mlu, 1)gpu"
    echo "|  eg. ./run_and_time.sh 0 0"
    echo "|      which means running from scratch using four MLU cards."
    echo "-------------------------------------------------------------"
}

if [[ $2 =~ ^[0-1]{1}$ ]]; then
    echo "Parameters Exact."
else
    echo "[ERROR] Unknow Parameter."
    usage
    exit 1
fi

THRESHOLD=1.0
if [ -z $PYTORCH_TRAIN_DATASET ]; then
    echo "[ERROR] Please set PYTORCH_TRAIN_DATASET."
    exit 1
fi
if [ -z $PYTORCH_TRAIN_CHECKPOINT ]; then
    echo "[ERROR] Please set PYTORCH_TRAIN_CHECKPOINT."
    exit 1
fi
BASEDIR=${PYTORCH_TRAIN_DATASET}
DATASET=${DATASET:-ml-20m}
ckp_dir=${CUR_DIR}/ckp
nproc_per_node=8
device='gpu'

# for mlu default using 8 cards
if [ $2 -eq 0 ]; then
    nproc_per_node=1
    device='mlu'
fi

# Get command line seed
seed=${1:-1}

# Get the multipliers for expanding the dataset
USER_MUL=${USER_MUL:-4}
ITEM_MUL=${ITEM_MUL:-16}

DATASET_DIR=${BASEDIR}/${DATASET}x${USER_MUL}x${ITEM_MUL}

if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"

	python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} --master_port 29501 ncf.py ${DATASET_DIR} \
        -l 0.0002 \
        -b 65536 \
        --layers 256 256 128 64 \
        -f 64 \
		--seed $seed \
        --threshold ${THRESHOLD} \
        --user_scaling ${USER_MUL} \
        --item_scaling ${ITEM_MUL} \
        --cpu_dataloader \
        --workers 8 \
        --random_negatives \
        --device $device \
        --do_train \
        --save_ckp 1 \
        --ckpdir ./ckp \
        --multiprocessing-distributed \
        --iters -1 \

	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="recommendation"


	echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"
else
	echo "Directory ${DATASET_DIR} does not exist"
fi
