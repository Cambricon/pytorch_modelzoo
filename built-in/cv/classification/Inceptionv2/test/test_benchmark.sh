#!/bin/bash
set -e

export IMAGENET_TRAIN_CHECKPOINT=/data/pytorch/models/pytorch_weight/checkpoints/

CUR_DIR=$(cd $(dirname $0);pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 net-precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             precision: fp32, O0, O1, O2, O3, amp"  
    echo "|             device: mlu, gpu"  
    echo "|             option1(multicards): ddp"  
    echo "|             option2(dummy test): dummy_test"  
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running inceptionv2 net on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh O1-mlu-ddp"
    echo "|      which means running inceptionv2 net on 4 MLU cards with O1 precision."
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
set_configs "$config"

# config配置到网络脚本的转换
main() {
    export DATASET_NAME="ImageNet_2012"
    pip install -r requirements.txt
    run_cmd="python src/train.py \
        --config configs/inception_v2/inception_v2_train_ddp_daily.yaml \
        --train-workers $num_workers \
        --train-batch-size $batch_size \
        --train-dataset $IMAGENET_TRAIN_DATASET/train \
        --resume $IMAGENET_TRAIN_CHECKPOINT/inception_v2/epoch_49.pth \
        --valid-dataset $IMAGENET_TRAIN_DATASET/val \
        --device $device \
        --train_iterations $train_iterations \
        $evaluation"

    # 配置DDP相关参数
    if [[ $ddp == "True" ]]; then
      export MASTER_ADDR='127.0.0.1'
      export MASTER_PORT=29500
      ddp_params="--distributed"
      if [[ $device == "gpu" ]]; then
        ddp_params="${ddp_params} --dist-backend nccl"
      else
        ddp_params="${ddp_params} --dist-backend cncl"
      fi
      run_cmd="${run_cmd} ${ddp_params}"
    fi

    # dummy test
    if [[ ${dummy_test} == "True" ]]; then
      run_cmd="$run_cmd --dummy_test"
    fi


    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      run_cmd="${run_cmd} --cnmix --opt_level ${precision} "
    elif [[ ${precision} == "pyamp" ]]; then
      run_cmd="${run_cmd} --pyamp"
    fi

    # 配置迭代次数
    if [[ $iters ]]; then
        run_cmd="${run_cmd} --iters ${iters}"
    fi

    # 参数配置完毕，运行脚本
    echo "cmd---------------------------------------"
    echo "$run_cmd"
    eval "${run_cmd}"
    echo "cmd---------------------------------------"
}


pushd ${CUR_DIR}/../models
main
popd
