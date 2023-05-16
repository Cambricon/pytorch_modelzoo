#!/bin/bash

CUR_DIR=$(cd $(dirname $0);pwd)

# help function
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 precision-device-[options...]"
    echo "|      Supported options:"
    echo "|         precision: fp32, amp"
    echo "|         device: mlu, gpu" 
    echo "|         option1(multicards): ddp"
    echo "|  eg.1 bash test_benchmark.sh fp32-mlu"
    echo "|      which means running ENet network on single MLU card with fp32 precision."
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh fp32-mlu-ddp"
    echo "|      which means running ENet net on 4 MLU cards with fp32 precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

config_file=""
while getopts 'h:' opt; do
   case "$opt" in
       h)  usage ; exit 1 ;;
       ?)  echo "unrecognized optional arg : "; $opt; usage; exit 1;;
   esac
done

config=$1
source ${CUR_DIR}/params_config.sh
set_configs "$config"

main() {
    export DATASET_NAME="CityScapes"
    run_cmd="python $use_launch main.py -m train    \
                   --epochs ${max_epochs} \
                   -b ${batch_size} \
                   --learning-rate ${learning_rate} \
                   --seed ${seed} \
                   --device $device \
                   --deterministic  \
                   --name enet  \
                   --dataset cityscapes \
                   --save-dir ./checkpoints    \
                   --dataset-dir $data  \
                   --height $height \
                   --width  $width  \
                   --learning-rate  $learning_rate  \
                   --lr-decay $lr_decay \
                   --dist-backend $dist_backend"

    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      echo "Not support pytorch cnmix yet, run precision fp32 instead."
    elif [[ ${precision} == "amp" ]]; then
      run_cmd="${run_cmd} --pyamp"
    fi

    if [[ $train_steps ]]; then
        run_cmd="${run_cmd} --iters ${train_steps}"
    fi

    if [[ $ddp == "True" ]]; then
        run_cmd="${run_cmd} --distributed"
    fi
    # 参数配置完毕，运行脚本
    if [ "$debug_mode" == "True" ] && [ -n "$CNNL_GEN_CASE" ]  && [ "$CNNL_GEN_CASE" -ne 0 ]; then
        echo "cmd---------------------------------------"
	echo "$run_cmd"
        eval "${run_cmd}"
    elif [ "$debug_mode" == "True" ]; then
        echo "cmd---------------------------------------"
	cnperf_path=$NEUWARE_HOME/bin/cnperf-cli
	echo "$run_cmd"
        eval "${cnperf_path} record ${run_cmd}"
        eval "${cnperf_path} timechart -o $CUR_DIR/../. --name ENET_timechart.json"
    else
        echo "cmd---------------------------------------"
	echo "$run_cmd"
	eval "${run_cmd}"
	echo "cmd---------------------------------------"
    fi
}


pushd ${CUR_DIR}/../models
pip install -r requirements.txt
if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"

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
