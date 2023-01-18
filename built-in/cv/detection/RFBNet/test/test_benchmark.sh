#!/bin/bash
set -e
# env
CUR_DIR=$(cd $(dirname $0);pwd)
RFBNet_DIR=$(cd ${CUR_DIR}/../models;pwd)

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
    echo "|      which means running RFBNet on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh -c O1-mlu-ddp"
    echo "|      which means running RFBNet on 4 MLU cards with O1 precision."
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
## 加载参数配置
config=$1
if [[ $config_file != "" ]]; then
    source $config_file
else
    source ${CUR_DIR}/params_config.sh
fi
set_configs "$config"

mkdir -p $PROJ_DIR/data/output/test_benchmark
rm -rf $PROJ_DIR/data/output/test_benchmark/*
rm -rf $PROJ_DIR/*.json
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT=28881
export OMP_NUM_THREADS=1 

# train cmd
run_cmd="RFBNet_train.py  \
        -d VOC -v RFB_vgg -s 300 \
        --device mlu \
        --batch_size ${batch_size} \
        --nprocessor ${card_num} \
        --save_folder $PROJ_DIR/data/output/test_benchmark \
        --mode scratch \
        -max $max_rounds \
        --basenet ${PYTORCH_TRAIN_CHECKPOINT}rfbnet/checkpoints_fp/vgg16_reducedfc.pth"

# infer cmd
check_cmd="RFBNet_infer.py -d VOC -v RFB_vgg -s 300 \
        --device mlu \
        --trained_model $PROJ_DIR/data/output/test_benchmark/Final_RFB_vgg_VOC.pth"

# config配置到网络脚本的转换
main() {
    pushd $RFBNet_DIR
    # 配置DDP相关参数
    if [[ $ddp == "True" ]]; then
      run_cmd="${run_cmd} --distributed --node_rank 0 --world_size 1 "
    fi

    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then  
      run_cmd="${run_cmd} --cnmix --opt_level ${precision} "
      check_cmd="$check_cmd --cnmix --opt_level ${precision}"
    elif [[ ${precision} == "pyamp" ]]; then
      run_cmd="${run_cmd} --pyamp"
      echo "Using AMP Train"
    fi

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
