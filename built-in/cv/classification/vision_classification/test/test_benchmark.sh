#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 net-precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             net: resnet18, resnet50, vgg16, mobilenet_v2,"
    echo "|                  alexnet, googlenet, resnet101, vgg19,"
    echo "|                  vgg16_bn, shufflenet_v2_x0_5, shufflenet_v2_x1_0,"
    echo "|                  shufflenet_v2_x1_5"
    echo "|             precision: fp32, O0, O1, O2, O3, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh resnet50-fp32-mlu"
    echo "|      which means running resnet50 net on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh resnet50-O1-mlu-ddp"
    echo "|      which means running resnet50 net on 4 MLU cards with O1 precision."
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

log_dir="${CUR_DIR}/../data/output/${net}_train_perf_log"
ckp_dir="${CUR_DIR}/../data/output/${net}_train_perf_ckps"

# config配置到网络脚本的转换
main() {
    run_cmd="python \
             $CUR_DIR/../classify_train.py \
             -a $net \
             --batch-size $batch_size \
             --lr $lr \
             --device $device \
             --momentum $momentum \
             --seed $seed \
             --data $IMAGENET_TRAIN_DATASET \
             --logdir $log_dir \
             --epochs $epochs \
             --save_ckp \
             --ckpdir $ckp_dir \
             --wd $weight_decay  \
             -j $num_workers "

    # 配置DDP相关参数
    if [[ $ddp == "True" ]]; then
      export MASTER_ADDR='127.0.0.1'
      export MASTER_PORT=29500
      ddp_params="--multiprocessing-distributed --world-size 1 --rank 0"
      if [[ $device == "gpu" ]]; then
        ddp_params="${ddp_params} --dist-backend nccl"
      else
        ddp_params="${ddp_params} --dist-backend cncl"
      fi
      run_cmd="${run_cmd} ${ddp_params}"
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

    # 配置resume参数
    if [[ ${resume} ]]; then
      run_cmd="$run_cmd --resume ${resume}"
      if [[ ${resume_multi_device} == "True" ]]; then
          run_cmd="$run_cmd --resume_multi_device"
      fi
    fi

    # 配置是否跑推理模式
    if [[ ${evaluate} == "True" ]]; then
        run_cmd="$run_cmd -e"
    fi

    # 参数配置完毕，运行脚本
    echo "$run_cmd"
    eval "${run_cmd}"
}


pushd $CUR_DIR
main
popd
