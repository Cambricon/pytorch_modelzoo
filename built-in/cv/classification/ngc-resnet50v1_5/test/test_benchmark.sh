#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
RES_DIR=$(cd ${CUR_DIR}/../models;pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 [config_file] net-precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             precision: fp32, O0, O1, O2, O3, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running ngc-resnet50v1_5 on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh O1-mlu-ddp"
    echo "|      which means running ngc-resnet50v1_5 net on 4 MLU cards with O1 precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

# 获取用户指定config函数并执行,得到对应config的参数配置
config_file=$OPTARG
while getopts 'h:' opt; do
   case "$opt" in
       h)  usage ; exit 1 ;;
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

train_script="python main.py"
optimizer_batch_size="2048"

run_cmd="$IMAGENET_TRAIN_DATASET  \
  --raport-file raport.json  \
  -p 100  \
  -j ${num_workers}    \
  --lr 2.048  \
  --warmup 8  \
  --arch resnet50  \
  -c fanin  \
  --label-smoothing 0.1  \
  --lr-schedule cosine  \
  --mom 0.875  \
  --wd 3.0517578125e-05  \
  --workspace ./ \
  -b $batch_size  \
  --seed 1  \
  --epochs $end_epoch  \
  --device $device  \
  --prof $prof  \
  --iters $iters  \
  --dist-backend $backend"

# config配置到网络脚本的转换
main() {
    export DATASET_NAME="ImageNet-2012"
    pushd $RES_DIR

    # 配置DDP相关参数
    if [[ $ddp == "True"  ]]; then
      train_script="python ./multiproc.py --nproc_per_node $DEVICE_COUNT main.py"
      optimizer_batch_size=$((2048 * $DEVICE_COUNT))
      if [[ $device == "gpu"  ]]; then
          backend="nccl"
      fi
    fi
    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      run_cmd="$run_cmd --cnmix --opt_level ${precision} "
    elif [[ ${precision} == "pyamp" ]]; then
      run_cmd="$run_cmd --pyamp"
    fi

    # 配置optimizer-batch-size，累计optimizer-batch-size的batch更新一次weight
    run_cmd="$run_cmd --optimizer-batch-size ${optimizer_batch_size}"

    # 配置resume参数
    if [[ ${resume} ]]; then
      run_cmd="$run_cmd --resume $IMAGENET_TRAIN_CHECKPOINT/resnet50v1_5/checkpoint.pth.tar "
    fi

    # 是否跑推理模式
    if [[ ${evaluate} == "False" ]]; then
        echo "Only train progress"
        run_cmd="$run_cmd --training-only"
    fi
    run_cmd="${train_script} ${run_cmd}"
    # 参数配置完毕，运行脚本
    echo "$run_cmd"
    eval "${run_cmd}"
    popd
}


pushd $CUR_DIR
export OMP_NUM_THREADS=1   # To avoid system being overloaded in multicard training process, we need to limit the value of OMP_NUM_THREADS
main
popd

