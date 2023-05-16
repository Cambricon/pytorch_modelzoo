#!/bin/bash
#set -e
# env
CUR_DIR=$(cd $(dirname $0);pwd)
SSD_ResNet50_DIR=$(cd ${CUR_DIR}/../models;pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 [-c] [config_file] precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             net: bert_msra"
    echo "|             precision: fp32, O0, O1, O2, O3, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh -c fp32-mlu"
    echo "|      which means running ssd_resnet50 on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh -c O1-mlu-ddp"
    echo "|      which means running ssd_resnet50 on 4 MLU cards with O1 precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

if [ -z ${COCO2017_TRAIN_DATASET} ]; then
  echo "please set environment variable COCO2017_TRAIN_DATASET."
  exit 1
fi

if [ -z ${PYTORCH_TRAIN_CHECKPOINT} ]; then
  echo "please set environment variable PYTORCH_TRAIN_CHECKPOINT."
  exit 1
fi
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

# train cmd
run_cmd="SSD_ResNet50_train.py  \
--backbone resnet50 \
--checkpoint $PYTORCH_TRAIN_CHECKPOINT/ssd/epoch_31.pt \
--backbone-path ${PYTORCH_TRAIN_CHECKPOINT}/ssd/resnet50-19c8e357.pth \
--bs ${batch_size} \
--seed $seed --epochs 33 --iterations $iters \
--device $device \
--save ./models --num-workers $num_workers \
--data $COCO2017_TRAIN_DATASET "

# infer cmd
check_cmd="SSD_ResNet50_infer.py \
        --backbone resnet50 \
        --backbone-path ${PYTORCH_TRAIN_CHECKPOINT}ssd/resnet50-19c8e357.pth \
	--checkpoint ./models/epoch_32.pt \
        --bs ${batch_size} --device $device \
        --mode evaluation --eval_iters $eval_iters \
        --data $COCO2017_TRAIN_DATASET "

# config配置到网络脚本的转换
main() {
    export DATASET_NAME="COCO2017"
    pushd $SSD_ResNet50_DIR
    pip install Cython==0.28.4
    pip install -r requirements.txt
    # 配置DDP相关参数
    if [[ $ddp == "True" ]]; then
      run_cmd="python -m torch.distributed.launch --nproc_per_node=$DEVICE_COUNT  $run_cmd --warmup 300"
    else
      run_cmd="python $run_cmd --warmup 32"
    fi

    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then  
      run_cmd="${run_cmd} --cnmix --opt_level ${precision} "
    elif [[ ${precision} == "amp" ]]; then
      run_cmd="${run_cmd} --pyamp"
      echo "Using AMP Train"
    fi

    if [[ ${cur_platform%_*} == "MLU590" ]]; then
	run_cmd="${run_cmd} --data-backend dali-mlu"
    fi

    # 参数配置完毕，运行脚本
    # To avoid system being overloaded in multicard training process, we need to limit the value of OMP_NUM_THREADS
    echo "$run_cmd"
    eval "$run_cmd"

    # R2
    if [[ ${evaluate} == "True" ]]; then
      echo "val_cmd: $check_cmd"
      eval "$check_cmd"
    fi
    popd
}

pushd $CUR_DIR
main
popd
