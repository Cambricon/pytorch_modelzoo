#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
WORK_DIR=$(cd ${CUR_DIR}/../models/;pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 net-precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             net: Tacotron2, WaveGlow"
    echo "|             precision: fp32, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh Tacotron2-fp32-mlu"
    echo "|      which means running Tacotron2 on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh Tacotron2-amp-mlu-ddp"
    echo "|      which means running Tacotron2 net on 4 MLU cards with AMP."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

if [ $# -lt 1 ]; then
   echo "[ERROR] Not enough arguments."
   usage
   exit 1
fi
configs=$1
# Paramaters check
if [[ ($configs =~ ^(Tacotron2|WaveGlow)-(fp32|amp)-(mlu|gpu)(-ddp)?(-ci.*)?) ]]; then
    echo "Paramaters Exact."
else
    echo "[ERROR] Unknow Parameter : " $configs
    usage
    exit 1
fi


## 加载参数配置
config=$1

source ${CUR_DIR}/params_config.sh

set_configs "$config"

if [ -z $MASTER_ADDR ]; then
    export MASTER_ADDR='127.0.0.100'
fi

if [ -z $MASTER_PORT ]; then
    export MASTER_PORT=29560
fi

log_dir=${WORK_DIR}/logs

# config配置到网络脚本的转换
main() {

    pushd $WORK_DIR

    mkdir -p $output
    
    train_cmd="$WORK_DIR/train.py -m ${net} \
      -o ${output} \
      -lr ${lr} \
      --epochs ${epochs} \
      -bs ${batch_size} \
      --weight-decay ${weight_decay} \
      --dist-url tcp://$MASTER_ADDR:$MASTER_PORT \
      --log-file ${log_file} \
      -d $PYTORCH_TRAIN_DATASET/TTS \
      --seed ${seed} \
      --cudnn-enabled"
      
    #使用MLU还是GPU
    if [[ $net == "Tacotron2" ]]; then
      train_cmd="${train_cmd} --grad-clip-thresh ${grad_clip_thresh} --anneal-steps $anneal_steps \
      --anneal-factor $anneal_factor"
    else
      if [[ ${precision} == "pyamp" ]]; then
        train_cmd="${train_cmd} --grad-clip-thresh ${grad_clip_thresh_amp}"
      else
        train_cmd="${train_cmd} --grad-clip-thresh ${grad_clip_thresh}"
      train_cmd="${train_cmd} --cudnn-benchmark \
      --segment-length=$segment_length \
       --cudnn_benchmark"
      fi
    fi
      
    if [[ $use_mlu == "True" ]]; then
      train_cmd="${train_cmd} --use-mlu"
    fi
    
    if [[ $cudnn_deterministic == "True" ]]; then
      train_cmd="${train_cmd} --cudnn-deterministic"
    fi

    # #配置ddp参数
    if [[ $ddp == "True" ]]; then
      if [[ $use_mlu == "True" ]]; then
        train_cmd="-m torch.distributed.launch --nproc_per_node=${num_cards} ${train_cmd} --dist-backend cncl"
      else
        train_cmd="-m multiproc --dist-backend nccl ${train_cmd} --dist-backend nccl"
      fi
    fi

    # 配置混合精度相关参数
    if [[ ${precision} == "pyamp" ]]; then
      train_cmd="$train_cmd --pyamp"
    elif [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      echo "Tacotron2 network not suppot CNMIX, Try AMP or FP32 instead"
      exit 1
    fi

    # 配置循环次数
    if [[ $iters ]]; then
      train_cmd="${train_cmd} --iter ${iters}"
    fi

    train_cmd="python ${train_cmd}"
    
    # 参数配置完毕，运行脚本
    echo "cmd---------------------------------------"
    echo "$train_cmd"
    eval "${train_cmd}"
    echo "cmd---------------------------------------"

    # # 是否跑推理模式
    # if [[ ${evaluate} == "True" ]]; then
    #     echo "${run_cmd}"
    #     eval "${run_cmd}"
    # fi
    popd   
}


pushd $CUR_DIR
main
popd
