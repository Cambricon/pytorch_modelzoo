#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             precision: fp32, O0, O1, O2, O3, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running yolov5s net on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh O1-mlu-ddp"
    echo "|      which means running yolov5s net on 4 MLU cards with O1 precision."
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

# config配置到网络脚本的转换
main() {
    train_cmd="python \
               train.py \
               --epochs $epochs \
               --data $data \
               --cfg $cfg \
               --device $device \
               --notest \
               --workers $num_workers \
               --batch-size $batch_size"

    test_cmd="python \
              test.py \
              --data $data \
              --weights $weights \
              --device $device"

    # 配置DDP相关参数
    if [[ $ddp == "True" ]]; then
      export MASTER_ADDR="127.0.0.1"
      export MASTER_PORT="8812"
      ddp_params="--multiprocessing-distributed --world-size 1 --rank 0"
      if [[ $device == "gpu" ]]; then
        ddp_params="${ddp_params} --dist-backend nccl"
      else
        ddp_params="${ddp_params} --dist-backend cncl"
      fi
      train_cmd="${train_cmd} ${ddp_params}"
    fi

    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      train_cmd="${train_cmd} --cnmix --opt_level ${precision} "
      test_cmd="${test_cmd} --cnmix --opt_level ${precision} "
    elif [[ ${precision} == "pyamp" ]]; then
      train_cmd="${train_cmd} --pyamp"
    fi

    # 配置训练迭代次数
    if [[ $iters ]]; then
        train_cmd="${train_cmd} --iters ${iters}"
    fi

    # 配置推理迭代次数
    if [[ $eval_iters ]]; then
        test_cmd="${test_cmd} --eval-iters ${eval_iters}"
    fi

    # 配置训练resume参数
    if [[ ${resume} == "True" ]]; then
      train_cmd="$train_cmd --resume --weights ${origin_weight} --weights-ema ${ema_weight}"
    fi

    if [[ ${evaluate} == "False" ]]; then
      # 运行训练脚本
      echo "cmd---------------------------------------"
      echo "$train_cmd"
      eval "${train_cmd}"
      echo "cmd---------------------------------------"
    else
      # 运行推理脚本
      echo "cmd---------------------------------------"
      echo "$test_cmd"
      eval "${test_cmd}"
      echo "cmd---------------------------------------"
    fi
}

pushd $CUR_DIR/../models/
if [ -z $PYTORCH_TRAIN_DATASET ]; then
  echo "[ERROR] Please set PYTORCH_TRAIN_DATASET."
  exit 1
fi

if [ -z $PYTORCH_TRAIN_CHECKPOINT ]; then
  echo "[ERROR] Please set PYTORCH_TRAIN_CHECKPOINT."
  exit 1
fi

# create datasets dir soft link
if [ ! -d "../coco" ]; then
    ln -sf "$PYTORCH_TRAIN_DATASET/COCO2017" "../coco"
fi
main
popd
