#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
DLRM_DIR=$(cd ${CUR_DIR}/../models/recommendation/pytorch;pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             precision: fp32, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running DLRM net on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh fp32-mlu-ddp"
    echo "|      which means running DLRM net on 4 MLU cards with fp32 precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

# 获取用户指定config函数并执行,得到对应config的参数配置
config_file=""
while getopts 'h:' opt; do
   case "$opt" in
       h)  usage ; exit 1 ;;
       ?)  echo "unrecognized optional arg : "; $opt; usage; exit 1;;
   esac
done
## 加载参数配置
config=$1

source ${CUR_DIR}/params_config.sh
source ${CUR_DIR}/../../../../tools/utils/common_utils.sh

set_configs "$config"

# default is single card
train_script="ncf.py"
use_launch=""

if [ -z ${PYTORCH_TRAIN_CHECKPOINT} ]; then
  echo "please set environment variable PYTORCH_TRAIN_CHECKPOINT."
  exit 1
fi

if [[ $ddp == "True" ]]; then
    if [ -z $MASTER_PORT ]; then
        export MASTER_ADDR='127.0.0.2'
    fi
    if [ -z $MASTER_PORT ]; then
        export MASTER_PORT=29500
    fi
fi


# config配置到网络脚本的转换
main() {
    export DATASET_NAME="MovieLens"
    pushd $DLRM_DIR
    pip install -r requirements.txt 
    # 配置卡数
    if [[ $ddp == "True" ]]; then
        get_visible_cards
        if [[ $cards_num -eq -1 ]]; then
          echo -e "\033[31m please set MLU_VISIBLE_DEVICES before run ddp. \033[0m"
          exit 1
        fi
        if [[ $cards_num -eq 16 ]]; then
            num_workers="7"
        fi
    else
        cards_num=1
	num_workers="16"
    fi

    echo "CARD_NUM is: " $cards_num

    # 配置DDP相关参数
    if [[ $ddp == "True" ]]; then
      use_launch="-m torch.distributed.launch --nproc_per_node=$cards_num --master_port $MASTER_PORT"
      train_script="ncf.py --multiprocessing-distributed"
    fi

    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      echo "Not support cnmix yet, run precision fp32 instead."
    elif [[ ${precision} == "amp" ]]; then
      train_script="${train_script} --use_amp 1"
    fi

    run_cmd="python $use_launch ${train_script} \
      --data ${DATASET_DIR}\
      --resume $PYTORCH_TRAIN_CHECKPOINT/DLRM/dlrm_19.pth \
      -l $lr \
      -b $batch_size \
      --layers $layers \
      -f $factors \
      --seed $seed \
      --save_ckp $save_ckp\
      --threshold $threshold \
      --user_scaling ${USER_MUL} \
      --item_scaling ${ITEM_MUL} \
      --cpu_dataloader \
      --random_negatives \
      --device $device \
      --do_train \
      --ckpdir $ckp_dir \
      --iters $iters \
      --workers $num_workers "

    # 是否跳过推理部分
    if [[ ${evaluate} == "True" ]]; then
      run_cmd="$run_cmd --do_predict --inference-iters ${inference_iters}"
    fi

    # dummy_test
    if [[ ${dummy_test} == "True" ]]; then
      run_cmd="$run_cmd --dummy_test"
    fi

    # 参数配置完毕，运行脚本
    echo "$run_cmd"
    eval "${run_cmd}"
    popd
}

pushd $CUR_DIR/../models/recommendation/pytorch/
pip install -r requirements.txt
pip install "git+https://github.com/mlperf/logging.git"
popd

pushd $CUR_DIR
main
popd
