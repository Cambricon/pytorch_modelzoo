#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             precision: fp32, O0, O1, O2, O3, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running pointpillar on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && bash test_benchmark.sh fp32-mlu-ddp"
    echo "|      which means running pointpillar on 8 MLU cards with fp32 precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

# 获取用户指定config函数并执行,得到对应config的参数配置
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

# config配置到网络脚本的转换
main() {
    export DATASET_NAME="nuscenes"
    if [[ ${evaluate} == "True" ]]; then
      run_cmd="python \
               -m torch.distributed.launch \
               --nproc_per_node=$card_num  \
               test.py \
               --launcher $launcher \
               --cfg_file $cfg_file \
               --ckpt $ckpt \
               --perf_max_iters $eval_iters"
    else
      run_cmd="python \
               -m torch.distributed.launch \
               --nproc_per_node=$card_num  \
               train.py \
               --launcher $launcher \
               --cfg_file $cfg_file \
               --fix_random_seed \
               --ckpt $ckpt \
               --epochs $total_epochs \
               --perf_max_iters $train_iters \
               --device $device"
    fi

    # 配置混合精度相关参数
    if [[ ${precision} == "amp" ]]; then
      echo "PointPillar have not supported AMP mode yet, please run in fp32 mode instead."
      exit 1
    elif [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      echo "PointPillar have not supported CNMIX mode yet, please run in fp32 mode instead."
      exit 1
    fi

    run_cmd="$run_cmd --set \
             OPTIMIZATION.BATCH_SIZE_PER_GPU $batch_size \
             OPTIMIZATION.LR $lr \
             OPTIMIZATION.LR_WARMUP True"

    # 参数配置完毕，运行脚本
    echo "cmd---------------------------------------"
    echo "$run_cmd"
    eval "$run_cmd"
    echo "cmd---------------------------------------"
}

# 编译pcdet包
pushd $CUR_DIR/../models
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install -r requirements.txt
python setup.py develop

# 清除历史训练数据
if [ -d "output" ]; then
    rm -rf "output"
fi

# 准备数据集目录
if [ -d "data/nuscenes" ]; then
    rm data/nuscenes
fi
ln -sf "${PYTORCH_TRAIN_DATASET}/nuScenes" "./data/nuscenes"

# 执行网络
pushd "tools/"
main

popd
popd
