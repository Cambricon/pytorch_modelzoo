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
    echo "|             net: BERT"
    echo "|             precision: fp32, O0, O1, O2, O3, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh O1-mlu-ddp"
    echo "|      which means running on 4 MLU cards with O1 precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

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

if [ -z $BERT_MODEL ]; then
    echo "[ERROR] Please set BERT_MODEL."
    exit 1
fi

if [ -z $SQUAD_DIR ]; then
    echo "[ERROR] Please set SQUAD_DIR."
    exit 1
fi
# default is single card
train_script="$BERT_DIR/models/scripts/run_squad.sh"
run_cmd="${train_script}  \
  $BERT_MODEL  \
  $epoch  \
  $batch_size  \
  $lr  \
  $precision  \
  $visible_cards  \
  1  \
  $SQUAD_DIR  \
  $BERT_DIR/models/checkpoints/squad/vocab.txt  \
  $BERT_DIR/models/output  \
  $mode  \
  $BERT_DIR/models/checkpoints/squad/bert_config.json  \
  $iters  \
  $eval_iters  \
  $hvd_cards \
  $opt_level \
  $device"


# config配置到网络脚本的转换
main() {

    pushd $BERT_DIR
    # 配置DDP相关参数
    if [[ $ddp == "True" ]]; then
      export MASTER_ADDR='127.0.0.1'
      export MASTER_PORT=29500
    fi

    # 参数配置完毕，运行脚本
    echo "$run_cmd"
    eval "${run_cmd}"
    popd
}


pushd $CUR_DIR
main
popd
