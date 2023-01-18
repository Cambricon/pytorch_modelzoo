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
    echo "|      which means running unet3d net on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh O1-mlu-ddp"
    echo "|      which means running unet3d net on 4 MLU cards with O1 precision."
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
    run_cmd="python $use_launch main.py --data_dir ${dataset_dir} \
                   --epochs ${max_epochs} \
                   --evaluate_every ${evaluate_every} \
                   --start_eval_at ${start_eval_at} \
                   --quality_threshold ${quality_threshold} \
                   --batch_size ${batch_size} \
                   --optimizer sgd \
                   --ga_steps ${gradient_accumulation_steps} \
                   --learning_rate ${learning_rate} \
                   --seed ${seed} \
                   --lr_warmup_epochs ${lr_warmup_epochs} \
                   --train_steps ${train_steps} \
                   --eval_steps ${eval_steps}"

    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      echo "Not support pytorch cnmix yet, run precision fp32 instead."
    elif [[ ${precision} == "amp" ]]; then
      run_cmd="${run_cmd} --amp"
    fi

    # 参数配置完毕，运行脚本
    echo "cmd---------------------------------------"
    echo "$run_cmd"
    eval "${run_cmd}"
    echo "cmd---------------------------------------"

}


pushd ${CUR_DIR}/../models
if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"
echo "${CUR_DIR}/.."
# CLEAR YOUR CACHE HERE
  python -c "
from mlperf_logging.mllog import constants
from runtime.logging import mllog_event
mllog_event(key=constants.CACHE_CLEAR, value=True)"
  # Set dataset name
    dataset_name=("KiTS19")
    export DATASET_NAME=${dataset_name}

    echo "BENCHMARK_LOG is "$BENCHMARK_LOG
    echo "AVG_LOG is "$AVG_LOG

    usage
    main
    
    # end timing
    end=$(date +%s)
    end_fmt=$(date +%Y-%m-%d\ %r)
    echo "ENDING TIMING RUN AT $end_fmt"


    # report result
    result=$(( $end - $start ))
    result_name="image_segmentation"


    echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"
else
    echo "Directory ${dataset_dir} does not exist"
fi
popd

