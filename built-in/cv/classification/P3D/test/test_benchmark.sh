#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

usage() {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 precision-device-[options...]"
    echo "|      Supported options:"
    echo "|         precision: fp32, O0, O1, O2, O3"
    echo "|         device: mlu, gpu" 
    echo "|         option1(multicards): ddp"
    echo "|  eg.1 bash test_benchmark.sh fp32-mlu"
    echo "|      which means running P3D NET on single MLU card with fp32 precision."
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh O0-mlu-ddp"
    echo "|      which means running P3D net on 4 MLU cards with O0 precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

while getopts 'h:' opt; do
   case "$opt" in
       h)  usage ; exit 1 ;;
       ?)  echo "unrecognized optional arg : "; $opt; usage; exit 1;;
   esac
done

# Checkout envs
if [ -z $PYTORCH_TRAIN_DATASET ]; then
    echo "[ERROR] Please set PYTORCH_TRAIN_DATASET."
    exit 1
fi

## 加载参数配置
config=$1
source ${CUR_DIR}/params_config.sh
set_configs "$config"

# start train network
main() {
    export DATASET_NAME="UCF101"
    run_cmd="$CUR_DIR/../models/main.py \
             $dataset \
             --train_steps $train_steps \
             --eval_steps ${eval_steps} \
             --batch-size $batch_size \
             --lr $lr \
             --logdir $log_dir \
             --device_param $device \
             --dropout ${dropout} \
             --seed ${seed} \
             --epochs $epochs \
             --print-freq=$print_freq \
             --dist-backend $backend  \
             --num-dev ${device_number}  \
             --workers $num_workers  "

    # 配置DDP相关参数
    if [[ $ddp == "True" ]]; then
      export MASTER_ADDR='127.0.0.1'
      export MASTER_PORT=29500
      use_ddp="-m torch.distributed.launch --master_port 29502 --nproc_per_node=${device_number}"
      if [[ $device == "gpu" ]]; then
        ddp_params="--dist-backend nccl"
      else
        ddp_params="--dist-backend cncl"
      fi
      run_cmd="${use_ddp} ${run_cmd} ${ddp_params}"
      export WORLD_SIZE=${device_number}
    fi

    run_cmd="python $run_cmd"  

    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      run_cmd="$run_cmd --cnmix --opt_level ${precision} "
    elif [[ ${precision} == "amp" ]]; then
      run_cmd="${run_cmd} --pyamp" 
    fi

    # dummy_test
    if [[ ${dummy_test} == "True" ]]; then
      run_cmd="$run_cmd --dummy_test"
    fi

    if [[ ${benchmark_mode} == True ]]; then
      echo "$run_cmd"
      eval "$run_cmd "
    else
      echo "$run_cmd --resume $resume"
      eval "$run_cmd --resume $resume"
    fi
}

pushd $CUR_DIR
main
popd
