#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
WAVERNN_DIR=$(cd ${CUR_DIR}/../models;pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|      Supported options:"
    echo "|             precision: fp32, O0, O1, O2, O3, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh  fp32-mlu"
    echo "|      which means running WaveRNN on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh O1-mlu-ddp"
    echo "|      which means running WaveRNN net on 4 MLU cards with O1 precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

if [ -z $PYTORCH_TRAIN_DATASET ]; then
    echo "[ERROR] Please set PYTORCH_TRAIN_DATASET."
    exit 1
fi

if [ -z $PYTORCH_TRAIN_CHECKPOINT ]; then
    echo "[ERROR] Please set PYTORCH_TRAIN_CHECKPOINT."
    exit 1
fi


# 获取用户指定config函数并执行,得到对应config的参数配置
config_file=""
while getopts 'hc:' opt; do
   case "$opt" in
       h)  usage ; exit 1 ;;
       c)  config_file=$OPTARG ;;
       ?)  echo "unrecognized optional arg : "; $opt; usage; exit 1;;
   esac
done
## 加载参数配置
config=$1
source ${CUR_DIR}/params_config.sh
source ${CUR_DIR}/../../../../tools/utils/common_utils.sh
set_configs "$config"

# Set dataset name
dataset_name="LJSpeech-1.1"
export DATASET_NAME=$dataset_name

if [[ $ddp == "True" ]]; then
    if [ -z $MASTER_PORT ]; then
        export MASTER_PORT=23456
    fi
fi

DEVIVE=$(echo $device | tr '[a-z]' '[A-Z]')

#default is single and daily
train_script="$WAVERNN_DIR/train.py --device $DEVIVE"

# config配置到网络脚本的转换

main() {
    pushd $WAVERNN_DIR
    pip install -r requirements.txt

    # create datasets dir
    if [ ! -d "./dataset" ]; then
        mkdir './dataset'
    fi

    # create datasets dir soft link
    if [ ! -d "./dataset/LJSpeech-1.1" ]; then
        ln -sf "$PYTORCH_TRAIN_DATASET/TTS/LJSpeech-1.1" "./dataset/LJSpeech-1.1"
    fi

    #data preprocess
    if [ ! -d "./dataset/data" ]; then
        python preprocess.py
    fi
    if [[ $ddp == "True" ]]; then
        get_visible_cards
        if [[ $cards_num -eq -1 ]]; then
          echo -e "\033[31m please set MLU_VISIBLE_DEVICES before run ddp. \033[0m"
          exit 1
        fi
        num_workers="12"
    else
        cards_num=1
    fi
    echo "CARD_NUM is: " $cards_num
    echo "precision is " $precision
    if [[ $ddp == "True" ]]; then
      train_script="${train_script} --dist-url $MASTER_PORT"
      use_launch="-m torch.distributed.launch --nproc_per_node=$cards_num"
    fi
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      echo "Not support cnmix yet, run precision fp32 instead."
    fi
    if [[ ${precision} == "pyamp" ]]; then
      train_script="${train_script} --amp"
    fi
    if [[ ${resume} == "True" ]]; then
      train_script="${train_script} --checkpoint-path $PYTORCH_TRAIN_CHECKPOINT/WaveRNN/checkpoint_WaveRNN_10.pt"
    fi
    if [[ ${num_per_checkpoint} -gt 0 ]]; then
      train_script="${train_script} --num-per-checkpoint ${num_per_checkpoint}"
    fi
    run_cmd="python $use_launch ${train_script} \
      --do-train \
      --num-workers $num_workers \
      --epochs $epochs \
      --lr $lr \
      --batch-size $batch_size \
      --seed $seed \
      --iterations $iters"
    #是否跑推理模式
    if [[ ${evaluate} == "True" ]]; then
      run_cmd="$run_cmd --eval ${iters}"
    fi
    # 参数配置完毕，运行脚本
    echo "cmd---------------------------------------"
    echo "$run_cmd"
    eval "${run_cmd}"
    popd
}

pushd $CUR_DIR
main
popd

