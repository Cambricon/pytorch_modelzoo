#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
CRF_DIR=$(cd ${CUR_DIR}/../models;pwd)
echo "*****"
echo $CRF_DIR
# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash [config_file] precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             net: CRF"
    echo "|             precision: fp32, O0, O1, O2, O3, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running on single MLU card with fp32 precision."
    echo "|                                                   "
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

# 获取用户指定config函数并执行,得到对应config的参数配置
config_file=$OPTARG
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

if [ -z $CRF_DATASET ]; then
    echo "[ERROR] Please set CRF_DATASET."
    exit 1
fi

# pushd $CRF_DIR/; pip install -r requirements.txt; popd
run_cmd="python main.py --device $device --iters $iters --data $CRF_DATASET"
# config配置到网络脚本的转换
main() {
    pushd ${CRF_DIR}

    # 配置混合精度相关参数
    if [[ ${precision} =~ ^O[0-3]{1}$ ]]; then
        echo "Not support CNMIX yet, run precision fp32 instead."
        exit 1
    elif [[ ${precision} == "amp" ]]; then
        echo "Not support pytorch AMP yet, run precision fp32 instead."
        exit 1
    fi
    if [[ $ddp == "True" ]]; then
        echo "CRF don't support ddp or horovod"
        exit 1
    fi

    # 参数配置完毕，运行脚本
    echo "$run_cmd"
    eval "${run_cmd}"
    popd
}

pushd ${CUR_DIR}/..
echo "${CUR_DIR}/.."
main
popd
