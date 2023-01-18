#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
BERT_DIR=$(cd ${CUR_DIR}/../;pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 [config_file] precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             net: bert-base-cased"
    echo "|             precision: fp32, O0, O1, O2, O3, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running bert_msra on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh O1-mlu-ddp"
    echo "|      which means running bert_msra on 4 MLU cards with O1 precision."
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
source ${CUR_DIR}/params_config.sh
set_configs "$config"

if [[ $ddp == "True" ]]; then
    if [ -z $MASTER_ADDR ]; then
        export MASTER_ADDR='127.0.0.1'
    fi
    if [ -z $MASTER_PORT ]; then
        export MASTER_PORT=29500
    fi
fi


# config配置到网络脚本的转换
main() {
    run_cmd="python $use_launch run_squad.py \
        --model_type $model_type        \
        --model_name_or_path bert-base-cased    \
        --do_train      \
        --train_file ${train_file}      \
        --predict_file ${predict_file}  \
        --per_gpu_train_batch_size ${batch_size}        \
        --learning_rate ${lr}   \
        --num_train_epochs ${num_train_epochs}  \
        --max_seq_length ${max_seq_length}      \
        --doc_stride ${doc_stride}      \
        --max_steps ${train_iters}      \
        --eval_iters ${eval_iters}  \
        --cache_dir ${bert_checkpoint}  \
        --output_dir ${output_dir}      \
        --overwrite_output_dir  \
        --device_param ${device_param}"

    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      run_cmd="${run_cmd} --cnmix --fp16_opt_level ${precision} "
    elif [[ ${precision} == "pyamp" ]]; then
      run_cmd="${run_cmd} --fp16_opt_level amp --amp "
    fi

    # 参数配置完毕，运行脚本
    echo "$run_cmd"
    eval "${run_cmd}"
}

bert_checkpoint=~/.cache/torch
if [ ! -d "${bert_checkpoint}/transformers" ];then
  if [ ! -d ${bert_checkpoint} ];then
    mkdir $bert_checkpoint
  fi  
  src="$IMAGENET_TRAIN_CHECKPOINT/bert-base-cased/torch"
  ln -s $src/transformers ${bert_checkpoint}/transformers
fi

pushd $CUR_DIR/../models
main
popd
