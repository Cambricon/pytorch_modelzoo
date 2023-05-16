#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
TRANS_DIR=$(cd ${CUR_DIR}/../models/;pwd)

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
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running bert_msra net on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh O1-mlu-ddp"
    echo "|      which means running bert_msra net on 4 MLU cards with O1 precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

CORPORA_PATH=${TRANS_DIR}/corpora
CKPT_MODEL_PATH=${TRANS_DIR}/ckpt_model
if [ -z ${IWSLT_CORPUS_PATH} ]; then
  echo "please set environment variable IWSLT_CORPUS_PATH."
  exit 1
fi
if [ -z ${TRANSFORMER_CKPT} ]; then
  echo "please set environment variable TRANSFORMER_CKPT."
  exit 1
fi

if [ ! -d ${CORPORA_PATH} ]; then
  ln -s ${IWSLT_CORPUS_PATH} ${CORPORA_PATH}
fi
if [ ! -d ${CKPT_MODEL_PATH} ]; then
  ln -s ${TRANSFORMER_CKPT} ${CKPT_MODEL_PATH}
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

log_dir=${TRANS_DIR}/logs

run_cmd="train.py  \
  --log-path ${log_dir} \
  --num_epochs ${num_epochs} \
  --iterations ${iters} \
  --print-freq 1 \
  --device ${device} \
  --batch-size ${batch_size} \
  --workers ${num_workers} \
  --dropout_rate 0.0"

check_cmd="eval.py \
--pretrained ${TRANS_DIR}/models/model_epoch_10.pth \
--device $device"

# config配置到网络脚本的转换
main() {
    export DATASET_NAME="IWSLT2016"
    pushd $TRANS_DIR
    pip install -r requirements.txt
    # 配置DDP相关参数
    if [[ $ddp == "True" ]]; then
      run_cmd="$run_cmd --distributed"
    fi

    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      run_cmd="${run_cmd} --cnmix --opt_level ${precision} "
      check_cmd="$check_cmd --cnmix --opt_level ${precision}"
    elif [[ ${precision} == "amp" ]]; then
      echo "Not support pytorch AMP yet, run precision fp32 instead."
    fi

    if [[ ${resume} == "True" ]]; then
      run_cmd="$run_cmd --resume ${CKPT_MODEL_PATH}/model_epoch_09.pth"
    fi

    rm -rf ${log_dir}
    # 参数配置完毕，运行脚本
    # To avoid system being overloaded in multicard training process, we need to limit the value of OMP_NUM_THREADS
    echo "$run_cmd"
    eval "OMP_NUM_THREADS=1 python $run_cmd"

    # R2
    if [[ ${evaluate} == "True" ]]; then
      echo $check_cmd
      eval "python $check_cmd"
    fi

    popd
}


pushd $CUR_DIR
main
popd
