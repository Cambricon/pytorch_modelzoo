#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
BERT_DIR=$(cd ${CUR_DIR}/../models/;pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 [config_file] precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             net: bert_msra"
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
#check env
if [ -z ${PYTORCH_TRAIN_CHECKPOINT} ]; then
  echo "please set environment variable PYTORCH_TRAIN_CHECKPOINT."
  exit 1
fi
# 获取用户指定config函数并执行,得到对应config的参数配置
config_file=$OPTARG
while getopts 'hc:' opt; do
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

# Set dataset name
dataset_name="Chinese_NER_data_MSRA"
export DATASET_NAME=$dataset_name

# default is single card
train_script="train.py"
use_launch=""

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

    pushd $BERT_DIR
    pip install -r requirements.txt
    # 配置DDP相关参数
    if [[ $ddp == "True" ]]; then
      use_launch="-m torch.distributed.launch --nproc_per_node=${nproc_per_node} --nnodes=1 --node_rank=0"
      train_script="train_ddp.py --distributed --nproc_per_node ${nproc_per_node}"
      if [[ $device == "gpu" ]]; then
        train_script="$train_script --dist-backend nccl"
      else
        train_script="$train_script --dist-backend cncl"
      fi
    fi

    run_cmd="python $use_launch ${train_script}  \
      --bert_model_dir $PYTORCH_TRAIN_CHECKPOINT/bert-base-chinese-pytorch  \
      --run_epochs $run_epochs  \
      --iters $iters  \
      --batch_size $batch_size  \
      --eval_iters $eval_iters  \
      --device $device"

    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      run_cmd="${run_cmd} --cnmix --opt_level ${precision} "
    elif [[ ${precision} == "pyamp" ]]; then
      run_cmd="${run_cmd} --pyamp"
    fi

    # 参数配置完毕，运行脚本
    echo "$run_cmd"
    eval "${run_cmd}"
    popd
}


pushd $CUR_DIR
main
popd
