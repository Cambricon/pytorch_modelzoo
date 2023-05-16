#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             precision: fp32"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|             option2(dummy test): dummy_test"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running crnn net on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh O1-mlu-ddp"
    echo "|      which means running crnn net on 4 MLU cards with O1 precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

# 获取用户指定config函数并执行,得到对应config的参数配置
config_file=""
while getopts 'h:' opt; do
   case "$opt" in
       h)  usage ; exit 1 ;;
       ?)  echo "unrecognized optional arg : "; $opt; usage; exit 1;;
   esac
done
## 加载参数配置
config=$1
echo "printing config"
source ${CUR_DIR}/params_config.sh
set_configs "$config"

log_dir="${CUR_DIR}/../${net}_one_card_log"
ckp_dir="${CUR_DIR}/../${net}_one_card_ckps"
org_dir="${CUR_DIR}/../"

pushd $org_dir/models
pip install -r requirements.txt
popd

# config配置到网络脚本的转换
main() {
    export DATASET_NAME="Synth90k"
    run_cmd="python train.py \
	     --adam \
             --lr $lr \
             --beta1 $beta\
             --trainRoot $data_path \
             --valRoot $data_path \
             --batchSize $batch_size \
             --nepoch $epoch \
             --displayInterval $display_interval \
             --workers $num_workers \
             --cudnn_lstm"
    if [[ ${dummy_test} == "True" ]]; then
        run_cmd="$run_cmd --dummy_test"
    fi

    # 配置设备相关参数
    if [[ $device == "mlu" ]]; then
      run_cmd="${run_cmd} --mlu"
    else
      run_cmd="${run_cmd} --cuda"
    fi

    # 配置DDP相关参数
    if [[ $ddp == "True" && $visible_cards -ne -1 ]]; then
        run_cmd="${run_cmd} --ddp True --ngpu $visible_cards"
    else
        run_cmd="${run_cmd} --ngpu 1"
    fi

    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
      run_cmd="${run_cmd} --cnmix --opt_level ${precision} "
    fi

    # 配置迭代次数
    if [[ $iters ]]; then
        run_cmd="${run_cmd} --iter ${iters}"
    fi

    # 配置resume参数
    if [[ ${resume} ]]; then
      run_cmd="$run_cmd --pretrained ${resume}"
    fi

    # 参数配置完毕，运行脚本
    echo "cmd---------------------------------------"
    echo "$run_cmd"
    eval "${run_cmd}"
    echo "cmd---------------------------------------"
}


pushd $CUR_DIR/../models
main
popd

