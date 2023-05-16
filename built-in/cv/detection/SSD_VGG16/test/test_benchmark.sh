#!/bin/bash
set -e
# env
CUR_DIR=$(cd $(dirname $0);pwd)
SSD_VGG16_DIR=$(cd ${CUR_DIR}/../models;pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 [-c] [config_file] precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             net: ssd_vgg16"
    echo "|             precision: fp32, O0, O1, O2, O3, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh -c fp32-mlu"
    echo "|      which means running  on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh -c O1-mlu-ddp"
    echo "|      which means running on 4 MLU cards with O1 precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

if [ -z ${PYTORCH_TRAIN_CHECKPOINT} ]; then
  echo "please set environment variable PYTORCH_TRAIN_CHECKPOINT."
  exit 1
fi
if [ -z ${PYTORCH_TRAIN_DATASET} ]; then
  echo "please set environment variable PYTORCH_TRAIN_DATASET."
  exit 1
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


# train cmd
run_cmd="SSD_VGG16_train.py --dataset_root ${PYTORCH_TRAIN_DATASET}/VOCdevkit \
	--start_iter 60000 \
        --dataset VOC --seed 42 --iters ${iters} --device $device \
        --batch_size ${batch_size} --lr $lr \
	--resume ${PYTORCH_TRAIN_CHECKPOINT}/ssd_vgg16/checkpoints_fp/ssd300_COCO_60000.pth "

# infer cmd
check_cmd="SSD_VGG16_test.py --trained_model ./mlu_weights/ssd300_VOC_60002.pth \
          --voc_root ${PYTORCH_TRAIN_DATASET}/VOCdevkit \
          --device mlu \
          --eval_iters ${eval_iters} "

# config配置到网络脚本的转换
main() {
    export DATASET_NAME="VOC0712"
    pushd $SSD_VGG16_DIR
    pip install Cython==0.28.4
    pip install -r requirements.txt
    # 配置DDP相关参数
    if [[ $ddp == "True" ]]; then
      run_cmd="$run_cmd --world-size 1 --rank 0 --multiprocessing-distributed --dist-backend cncl"
    fi

    # 配置混合精度相关参数
    if [[ ${precision} != "fp32" ]]; then  
        if [[ $precision == "amp" ]]; then
	    run_cmd="${run_cmd} --pyamp"
        elif [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
            run_cmd="${run_cmd} --cnmix --opt_level ${precision}"
        else
	    echo "Unsupported precision. "
	fi
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
