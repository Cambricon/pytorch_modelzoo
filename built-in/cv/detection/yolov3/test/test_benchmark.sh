#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
TRANS_DIR=$(cd ${CUR_DIR}/../models/;pwd)

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
    echo "|      which means running yolov3 net on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh O1-mlu-ddp"
    echo "|      which means running yolov3 net on 4 MLU cards with O1 precision."
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
    pushd $TRANS_DIR
    train_cmd="python \
              $CUR_DIR/../models/train.py \
              --notest \
              --cfg $cfg \
              --data $data \
              --ckp-path $ckp_dir \
              --logdir $log_dir \
              --device $device \
              --batch-size $batch_size \
              --epochs $epochs \
              --dist-url 'tcp://127.0.0.1:1991' \
              --workers $num_workers"

    test_cmd="python \
              $CUR_DIR/../models/test.py \
              --img-size $img_size \
              --cfg $cfg \
              --data $data \
              --iou-thr $iou \
              --task $task \
              --weights $weights \
              --device $device \
              --batch-size $batch_size \
              "

    # 配置DDP相关参数
    if [[ $ddp == "True" ]]; then
      export MASTER_ADDR="127.0.0.1"
      export MASTER_PORT="29400"
      ddp_params="--distributed --world-size 1 --rank 0"
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
      train_cmd="$train_cmd --resume --weights ${weights}"
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

    popd
}

pushd $CUR_DIR
main
popd
