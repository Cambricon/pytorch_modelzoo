#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)
NET_ROOT_DIR=$(cd ${CUR_DIR}/../models/;pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0  [config_file] net-precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             precision: fp32"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh -c fp32-mlu"
    echo "|      which means running DeepSpeech2 on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh fp32-mlu-ddp"
    echo "|      which means running DeepSpeech2 net on 4 MLU cards with fp32 precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

# 获取用户指定config函数并执行,得到对应config的参数配置
while getopts 'h:' opt; do
   case "$opt" in
       h)  usage ; exit 1 ;;
       ?)  echo "unrecognized optional arg : "; $opt; usage; exit 1;;
   esac
done
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

# config配置到网络脚本的转换
main() {
    pushd $NET_ROOT_DIR/pytorch
    # ddp_params=""
    # 安装依赖库
    pip install -r $NET_ROOT_DIR/requirements.txt
    save_folder="./models"
    model_path="./models/deepspeech2_final.pth.tar"
    run_cmd="train.py \
             --device $device \
             --save_folder $save_folder \
             --model_path  $model_path \
             --num_workers $num_workers \
             --checkpoint               \
             --iters       $iters \
             --eval_iters  $eval_iters"
    if [[ $ddp == "True" ]];then
        ddp_params="-m torch.distributed.launch --nproc_per_node=${cards_num}"
    fi

    if [[ $resume == "True" ]];then
        export PYTORCH_TRAIN_CHECKPOINT=/data/pytorch/models/pytorch_weight/checkpoints/
        resume_params="--continue_from $PYTORCH_TRAIN_CHECKPOINT/deepspeech2/epoch_9.pth.tar"
    fi

    run_cmd="python ${ddp_params} ${run_cmd} ${resume_params}"
    # 运行训练脚本
    echo "---------------------------------"
    echo "$run_cmd"
    eval "$run_cmd"
    echo "---------------------------------"
    popd

    # # 是否跑推理模式
    # if [[ ${evaluate} == "True" ]]; then
    #     echo "${run_cmd}"
    #     eval "${run_cmd}"
    # fi
    # popd
}


pushd $CUR_DIR
if [ ! -d "../models/LibriSpeech_dataset" ]; then
  ln -s $PYTORCH_TRAIN_DATASET/LibriSpeech_dataset  ../../
fi
main
popd
