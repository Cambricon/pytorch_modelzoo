#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
OLTR_DIR=$(cd ${CUR_DIR}/../models/;pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 [-c] [config_file] net-precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             precision: fp32, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running OLTR on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh fp32-mlu-ddp"
    echo "|      which means running OLTR net on 4 MLU cards with fp32 precision."
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

if [ -z $IMAGENET_TRAIN_CHECKPOINT ]; then
    echo "[ERROR] Please set IMAGENET_TRAIN_CHECKPOINT."
    exit 1
fi

## 加载参数配置
config=$1
if [[ $config_file != "" ]]; then
    source $config_file
else
    source ${CUR_DIR}/params_config.sh
fi
set_configs "$config"

if [ -z $MASTER_ADDR ]; then
    export MASTER_ADDR='127.0.0.100'
fi

if [ -z $MASTER_PORT ]; then
    export MASTER_PORT=29560
fi

# create datasets dir soft link
if [ ! -L "${OLTR_DIR}/data/ImageNet_LT"  ]; then
  pushd $OLTR_DIR
  ln -sf "$PYTORCH_TRAIN_DATASET/ImageNet_LT" "./data/ImageNet_LT"
  popd
fi

#create checkpoint dir soft link
if [ ! -L "${OLTR_DIR}/logs/ImageNet_LT/imagenet_mid_stage1/final_model_checkpoint.pth" ]; then
  pushd $OLTR_DIR
  ln -sf "$IMAGENET_TRAIN_CHECKPOINT/BBN/stage_1/final_model_checkpoint.pth" "./logs/ImageNet_LT/imagenet_mid_stage1/final_model_checkpoint.pth"
  popd
fi

train_script="python main_imagenet.py --device $runnable_cards"

# config配置到网络脚本的转换
main() {
    export DATASET_NAME="ImageNet-LT"
    pushd $OLTR_DIR
    pip install -r requirements.txt
    train_cmd="${train_script} --data_path $IMAGENET_TRAIN_DATASET --config imagenet_mid_stage_2_meta_embedding.py --iters ${iters} --seed 1 --num_workers ${num_workers} --batch_size ${batch_size}"
    run_cmd="${train_script} --data_path $IMAGENET_TRAIN_DATASET --config imagenet_mid_stage_2_meta_embedding.py --iters ${iters} --test_open --seed 1"

    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      train_cmd="$train_cmd --cnmix --opt_level ${precision} "
      run_cmd="$run_cmd --cnmix --opt_level ${precision} "
    elif [[ ${precision} == "amp" ]]; then
      echo "Not support pytorch AMP yet, run precision fp32 instead."
    fi

    # 配置resume参数
    if [[ ${resume} ]]; then
      train_cmd="$train_cmd --resume $IMAGENET_TRAIN_CHECKPOINT/BBN/stage_2/epoch_50.pth"
      run_cmd="$run_cmd --resume logs/ImageNet_LT/imagenet_mid_meta_embedding_exp2/final_model_checkpoint.pth"
    fi

    # 参数配置完毕，运行脚本
    echo "cmd---------------------------------------"
    echo "$train_cmd"
    eval "${train_cmd}"
    echo "cmd---------------------------------------"

    # 是否跑推理模式
    if [[ ${evaluate} == "True" ]]; then
        echo "${run_cmd}"
        eval "${run_cmd}"
    fi
    popd
}


pushd $CUR_DIR
main
popd
