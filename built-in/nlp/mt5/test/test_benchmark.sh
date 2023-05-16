#!/bin/bash
set -e

CUR_DIR=$(
  cd $(dirname $0)
  pwd
)
MODEL_DIR=$(
  cd ${CUR_DIR}/../models/
  pwd
)

# 帮助函数
function usage() {
  echo -e "\033[32m Usage : \033[0m"
  echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
  echo "|  bash $0 [-c] [config_file] precision-device-[options...]"
  echo "|      Supported options:"
  echo "|             net: mt5"
  echo "|             precision: fp32, amp"
  echo "|             device: mlu, gpu"
  echo "|             option1(multicards): ddp"
  echo "|                                                   "
  echo "|  eg.1. bash test_benchmark.sh -c fp32-mlu"
  echo "|      which means running mt5 net on single MLU card with fp32 precision."
  echo "|                                                   "
  echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh -c amp-mlu-ddp"
  echo "|      which means running mt5 net on 4 MLU cards with amp precision."
  echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

# 获取用户指定config函数并执行,得到对应config的参数配置
config_file=""
while getopts 'hc:' opt; do
  case "$opt" in
  h)
    usage
    exit 1
    ;;
  c) config_file=$OPTARG ;;
  ?)
    echo "unrecognized optional arg : "
    $opt
    usage
    exit 1
    ;;
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

log_dir=${MODEL_DIR}/logs

export PYTORCH_TRANSFORMERS_CACHE=True
python -m pip install -r ${CUR_DIR}/../requirements.txt
cp ${CUR_DIR}/../models/modeling_t5.py $(dirname $(python -c "import transformers; print(transformers.__file__)"))/models/t5/modeling_t5.py 
export MT5_SAVED_MODEL_DIR=${PWD}/saved_model
export MT5_SAVED_MODEL_NAME=summary_model
export DATASET_NAME=CSL
export MT5_CHECKPOINT_DIR=/data/AE/pytorch_eco/checkpoint/chinese_t5_pegasus_base
export CSL_DIR=/data/AE/pytorch_eco/datasets/CSL
test -d "$MT5_CHECKPOINT_DIR" || { echo "error: MT5_CHECKPOINT_DIR does not exist, please source .jenkins/net_env.sh"; exit 1; }
test -f "$CSL_DIR/benchmark/ts/train.tsv" || { echo "error: CSL_DIR/benchmark/ts/train.tsv does not exist, please source .jenkins/net_env.sh"; exit 1; }
test -f "$CSL_DIR/benchmark/ts/dev.tsv" || { echo "error: CSL_DIR/benchmark/ts/dev.tsv does not exist, please source .jenkins/net_env.sh"; exit 1; }

INFER_OUTPUT_DIR=$(realpath ${CUR_DIR}/../infer_output)
if [ ! -d ${INFER_OUTPUT_DIR} ]; then
  mkdir ${INFER_OUTPUT_DIR}
  echo "mkdir ${INFER_OUTPUT_DIR}"
fi


run_cmd="finetune.py  \
    --pretrain_model=${MT5_CHECKPOINT_DIR} \
    --train_data=${CSL_DIR}/benchmark/ts/train.tsv \
    --dev_data=${CSL_DIR}/benchmark/ts/dev.tsv \
    --saved_model_dir ${MT5_SAVED_MODEL_DIR} \
    --saved_model_name ${MT5_SAVED_MODEL_NAME} \
    --num_epoch ${num_epochs} \
    --train_iterations ${train_iters} \
    --valid_iterations ${valid_iters} \
    --eval_iterations=${eval_iters} \
    --device ${device} \
    --num_device ${nproc_per_node} \
    --batch_size ${batch_size}"


check_cmd="finetune.py \
    --pretrain_model=${MT5_CHECKPOINT_DIR} \
    --dev_data=${CSL_DIR}/benchmark/ts/dev.tsv \
    --saved_model_dir ${MT5_SAVED_MODEL_DIR} \
    --saved_model_name ${MT5_SAVED_MODEL_NAME} \
    --num_device 1 \
    --device MLU \
    --batch_size 4 \
    --eval_iterations=${eval_iters} \
    --mode evaluation"

# config配置到网络脚本的转换
main() {

  pushd $MODEL_DIR
  # 配置DDP相关参数
  if [[ $ddp == "True" ]]; then
    run_cmd="$run_cmd --distributed --num_device ${nproc_per_node}"
  fi

  # 配置混合精度相关参数
  if [[ ${precision} == "amp" ]]; then
    run_cmd="${run_cmd} --amp"
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
