#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
PIX2PIX_DIR=$(cd ${CUR_DIR}/../models/;pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 [-c] [config_file] net-precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             precision: fp32, O0, O1, O2, O3, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running Pix2pix on single MLU card with fp32 precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

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
if [[ $config_file != "" ]]; then
    source $config_file
else
    source ${CUR_DIR}/params_config.sh
fi
set_configs "$config"

# config配置到网络脚本的转换
main() {
    export DATASET_NAME="facades"
    pushd $PIX2PIX_DIR
    pip install -r requirements.txt

    trained_cmd="train.py --name facades_pix2pix_resnet_9blocks --model pix2pix --netG resnet_9blocks \
                --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --display_id 0 --seed 0 \
                --dataroot $PYTORCH_TRAIN_DATASET/facades --batch_size $batch_size --num_threads $num_workers \
                --iters $iters --device $device "

    test_cmd="test.py --name facades_pix2pix_resnet_9blocks --model pix2pix --netG resnet_9blocks \
              --direction BtoA --dataset_mode aligned --norm batch --eval \
              --dataroot $PYTORCH_TRAIN_DATASET/facades --device $device "

    # 配置DDP相关参数
    if [[ $ddp == "True"  ]]; then
      echo "Pix2pix have not support DDP mode yet, please run in single mode instead."
      exit 1
    fi

    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      echo "Pix2pix do not supported CNMIX, please run precision fp32."
      exit 1
    elif [[ ${precision} == "pyamp" ]]; then
      echo "Pix2pix do not supported AMP, please run precision fp32."
      exit 1
    fi

    # 配置resume参数
    if [[ ${resume} == "True" ]]; then
      trained_cmd="$trained_cmd --continue_train --epoch_count 100 --epoch 100 --resume_dir $resume_dir "
    fi

    # 参数配置完毕，运行脚本
    echo "cmd---------------------------------------"
    echo "python $trained_cmd"
    eval "python ${trained_cmd}"

    # 是否跑推理模式
    if [[ ${evaluate} == "True" ]]; then
        echo "python ${test_cmd}"
        eval "python ${test_cmd}"
    fi
    popd
}

pushd $CUR_DIR
main
popd
