#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
SWINT_DIR=$(cd ${CUR_DIR}/../models/;pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 [-c] [config_file] net-precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             precision: fp32, O0, O1, O2, O3, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running Swin-Transformer on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh amp-mlu-ddp"
    echo "|      which means running swin_transformer_ssl net on 4 MLU cards with amp precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

# 获取用户指定config函数并执行,得到对应config的参数配置
config_file=""
while getopts 'hc:' opt; do
   case "$opt" in
       h)  usage ; exit 1 ;;
       c)  config_file=$OPTARG ;;
       ?)  echo "unrecognized optional arg : "; $opt; usage; exit 1;;
   esac
done

# Checkout envs
if [ -z $IMAGENET_TRAIN_DATASET ]; then
    echo "[ERROR] Please set IMAGENET_TRAIN_DATASET."
    exit 1
fi

## 加载参数配置
config=$1
if [[ $config_file != "" ]]; then
    source $config_file
else
    source ${CUR_DIR}/params_config.sh
fi
set_configs "$config"

# Set dataset name
dataset_name="ImageNet2012"
export DATASET_NAME=$dataset_name

# config配置到网络脚本的转换
main() {

    pushd $SWINT_DIR

    pip install -r requirements.txt

    pretrained_cmd="moby_main.py --device=$device --cfg $YAML_CONFIG \
                                 --data-path $IMAGENET_TRAIN_DATASET --output $OUTPUT_DIR \
                                 --batch-size $batch_size --num-workers $num_workers \
                                 --iters $iters "

    trained_eval_cmd="moby_linear.py --device=$device --cfg $YAML_CONFIG \
                                 --data-path $IMAGENET_TRAIN_DATASET --output $OUTPUT_DIR \
                                 --batch-size $batch_size --num-workers $num_workers \
                                 --iters $iters --pretrained-ckpt $PRETRAINED_PTH "

    # 配置DDP相关参数
    if [[ $ddp == "True"  ]]; then
      pretrained_cmd="-m torch.distributed.launch --nproc_per_node $visible_cards --master_port $MASTER_PORT ${pretrained_cmd} --distributed "
      trained_eval_cmd="-m torch.distributed.launch --nproc_per_node $visible_cards --master_port $MASTER_PORT ${trained_eval_cmd} --distributed "
    fi

    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      pretrained_cmd="${pretrained_cmd} --apex --amp-opt-level ${precision} "
      trained_eval_cmd="${trained_eval_cmd} --apex --amp-opt-level ${precision} "
      if [[ ${device} == "mlu" && ${precision} -ne "O0" ]]; then
        echo "Swin transformer do not supported CNMIX, please run precision fp32 or AMP."
        exit 1
      fi
    elif [[ ${precision} == "pyamp" ]]; then
      pretrained_cmd="${pretrained_cmd} --pyamp "
      trained_eval_cmd="${trained_eval_cmd} --pyamp "
    fi

    # 配置resume参数
    if [[ ${resume} == "True" ]]; then
      pretrained_cmd="$pretrained_cmd --resume $RESUME_PRETRAINED"
      trained_eval_cmd="$trained_eval_cmd --resume $RESUME_LINEAR"
    fi

    trained_eval_cmd="${trained_eval_cmd} --eval_iters $eval_iters"

    # 参数配置完毕，运行脚本
    echo "cmd---------------------------------------"
    echo "python $pretrained_cmd"
    eval "python ${pretrained_cmd}"
    echo "cmd---------------------------------------"
    echo "python $trained_eval_cmd"
    eval "python ${trained_eval_cmd}"
    echo "cmd---------------------------------------"

    popd
}

pushd $CUR_DIR
main
popd
