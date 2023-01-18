#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
INC_DIR=$(cd ${CUR_DIR}/../models/;pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             precision: fp32, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running inception_v4 on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh amp-mlu-ddp"
    echo "|      which means running inception_v4 on 4 MLU cards with amp precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

# 获取用户指定config函数并执行,得到对应config的参数配置
while getopts 'h:' opt; do
   case "$opt" in
       h)  usage ; exit 1 ;;
       ?)  echo "unrecognized optional arg : "; $opt; usage; exit 1;;
   esac
done

# 检查环境变量
if [ -z $IMAGENET_TRAIN_DATASET ]; then
    echo "[ERROR] Please set IMAGENET_TRAIN_DATASET."
    exit 1
fi

## 加载参数配置
config=$1
source ${CUR_DIR}/params_config.sh
set_configs "$config"

# Set dataset name
dataset_name="ImageNet2012"
export DATASET_NAME=$dataset_name

# config配置到网络脚本的转换
main() {
    pushd $INC_DIR

    pip install -r requirements.txt

    run_cmd="train.py \
             $dataset \
             --model $model\
             --batch-size $batch_size \
             --iters $iters \
             --epochs $epochs \
             --opt $opt\
             --opt-eps $opt_eps\
             --lr $lr\
             --warmup-lr $warmup_lr\
             --weight-decay $weight_decay\
             --decay-rate $decay_rate\
             --decay-epochs $decay_epochs\
             --device $device\
             --cooldown-epoch $cooldown_epoch\
             --remode $remode\
             --reprob $reprob\
             --workers $num_workers \
             --output $output"

    # 配置Native Torch AMP相关参数
    if [[ ${precision} == "amp" ]]; then
       run_cmd="${run_cmd} --amp "
    fi

    if [[ ${model} == "inception_v4" ]]; then
        run_cmd="${run_cmd} \
                 --sched $sched \
                 --drop $drop \
                 --aa $aa \
                 --lr-noise $lr_noise"
    fi

    # 配置DDP相关参数
    if [[ $ddp == "True" ]]; then
      use_ddp="-m torch.distributed.launch --nproc_per_node=${visible_cards} --master_port $MASTER_PORT"
      run_cmd="${use_ddp} ${run_cmd}"
      export WORLD_SIZE=${visible_cards}
    fi

    run_cmd="python $run_cmd"

    # 参数配置完毕，运行脚本
    echo "cmd---------------------------------------"
    echo "$run_cmd"
    eval "${run_cmd}"
    echo "cmd---------------------------------------"
}

pushd $CUR_DIR
main
popd
