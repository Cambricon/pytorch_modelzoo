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
    echo "|  $0 [seed]"
    echo "|  parameter1: seed number, Seed 0 has been shown to converge deterministically."
    echo "|  eg. ./DLRM_FP32_20E_4MLUs_Train.sh 0"
    echo "|      which means running from scratch using seed 0."
    echo "-------------------------------------------------------------"
}

if [[ $1 =~ ^[0-5]{1}$ ]]; then
    echo "Parameters Exact."
else
    echo "[ERROR] Unknow Parameter."
    usage
    exit 1
fi

THRESHOLD=1.0
BASEDIR=${PYTORCH_TRAIN_DATASET}
DATASET=${DATASET:-ml-20m}
ckp_dir=${CUR_DIR}/ckp
nproc_per_node=4
device='mlu'

# Get command line seed
seed=${1:-1}

# Get the multipliers for expanding the dataset
USER_MUL=${USER_MUL:-4}
ITEM_MUL=${ITEM_MUL:-16}

DATASET_DIR=${BASEDIR}/${DATASET}x${USER_MUL}x${ITEM_MUL}

pushd $CUR_DIR/../models/recommendation/pytorch
pip install -r requirements.txt
pip install "git+https://github.com/mlperf/logging.git"

if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"

    export MLU_VISIBLE_DEVICES=0,1,2,3
	python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} --master_port 29501 ncf.py \
        --data ${DATASET_DIR} \
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

popd
