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
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running RFBNet on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh O1-mlu-ddp"
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

export MASTER_ADDR='127.0.0.1'
export MASTER_PORT=28881
export OMP_NUM_THREADS=1 

# train cmd
run_cmd="python RFBNet_train.py  \
        -d VOC -v RFB_vgg -s 300 \
	$resume_opt $resume_path \
	$distributed_flag --nprocessor $card_num \
	--world_size $world_size --node_rank $node_rank \
        --mode scratch --lr $lr \
	--num_workers $num_workers \
	--unit_in_iters \
        --batch_size ${batch_size} --max $iters \
        --device mlu \
        --save_folder $output "

# infer cmd
check_cmd="RFBNet_infer.py -d VOC -v RFB_vgg -s 300 \
        --device mlu \
        --trained_model $PROJ_DIR/data/output/test_benchmark/Final_RFB_vgg_VOC.pth"

# config配置到网络脚本的转换
main() {
    export DATASET_NAME="VOC2007"
    pushd $RFBNet_DIR
    pip install -r requirements.txt

    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then  
      run_cmd="${run_cmd} --cnmix --opt_level ${precision} "
    elif [[ ${precision} == "pyamp" ]]; then
      run_cmd="${run_cmd} --pyamp"
      echo "Using AMP Train"
    fi

    # 参数配置完毕，运行脚本
    echo "$run_cmd"
    eval "${run_cmd}"

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
