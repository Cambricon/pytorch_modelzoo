#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             precision: fp32,amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running swin-transformer on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh amp-mlu-ddp"
    echo "|      which means running swin-transformer on 4 MLU cards with native torch amp."
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

pushd $CUR_DIR/../models
pip install -r requirements.txt
popd

# config配置到网络脚本的转换
main() {
    export DATASET_NAME="ImageNet2012"
    run_cmd="$CUR_DIR/../models/train.py \
             $dataset \
             --model $model\
             --batch-size $batch_size \
             --iters $iters \
             --epochs $epochs \
             --opt $opt\
             --opt-eps $opt_eps\
             --opt-betas $opt_betas\
             --momentum $momentum\
             --lr $lr\
             --min-lr $min_lr\
             --warmup-epochs $warmup_epochs\
             --warmup-lr $warmup_lr\
             --weight-decay $weight_decay\
             --decay-rate $decay_rate\
             --decay-epochs $decay_epochs\
             --clip-grad $clip_grad\
             --device $device\
             --color-jitter $color_jitter\
             --cutmix $cutmix\
             --mixup $mixup\
             --mixup-mode $mixup_mode\
             --mixup-prob $mixup_prob\
             --recount $recount\
             --remode $remode\
             --reprob $reprob\
             --train-interpolation $train_interpolation\
             --seed $seed\
             --drop-path $drop_path\
             --pin-mem\
             --cooldown-epoch $cooldown_epoch\
             --output $output"

    # 配置Native Torch AMP相关参数
    if [[ ${precision} == "pyamp" ]]; then
       run_cmd="${run_cmd} --amp --native-amp "
    fi

    # 配置DDP相关参数
    if [[ $ddp == "True" ]]; then
      export MASTER_ADDR='127.0.0.1'
      export MASTER_PORT=29500
      use_ddp="-m torch.distributed.launch --nproc_per_node=${device_number}"
      run_cmd="${use_ddp} ${run_cmd}"
      export WORLD_SIZE=${device_number}
    fi

    run_cmd="python $run_cmd"

    # 参数配置完毕，运行脚本
    echo "cmd---------------------------------------"
    echo "$run_cmd"
    export PYTHONPATH="${CUR_DIR}/../../":$PYTHONPATH
    eval "${run_cmd}"
    echo "cmd---------------------------------------"
}

pushd $CUR_DIR
main
popd
