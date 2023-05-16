#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

base_params () {
    device="mlu"

    lr="0.0002"
    factors="64"
    layers="256 256 128 64"
    seed="0"
    threshold="1.0"
    BASEDIR=${PYTORCH_TRAIN_DATASET}
    DATASET=${DATASET:-ml-20m}
    batch_size="65536"
    save_ckp="1"
    precision="fp32"
    num_workers="8"
    ddp="False"
    ckp_dir=./ckp
    USER_MUL=${USER_MUL:-4}
    ITEM_MUL=${ITEM_MUL:-16}

    DATASET_DIR=${BASEDIR}/${DATASET}x${USER_MUL}x${ITEM_MUL}
    benchmark_mode="True"
    debug_mode="False"
    # max_batch_size_MLU290="30"
    # max_batch_size_MLU370="32"
    # max_batch_size_MLU590="32"
    # max_batch_size_V100="32"

}

set_configs () {
    # 调用相应网络的base_params
    base_params

    # 根据每个字段的功能, overide对应参数
    args=$1
    params_array=(${args//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)   ;;
            O[0-3]) precision=$var ;;
            amp)    precision="amp" ;;
            mlu)    ;;
            gpu)    device="gpu" ;;
            ddp)    ddp="True" ;;
     dummy_test)    dummy_test="True" ;;
      ci_train)     benchmark_mode=False;
		    debug_mode=True;
                    iters=2;
                    ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then
        ## 加载公用方法
        source ${CONFIG_DIR}/../../../../tools/utils/common_utils.sh

        ## 获取benchmark_mode计数规则,配置迭代数
        iters=-1
        perf_iters_rule iters

        device=$device
        ## 设置benchmark_mode log路径
        #export BENCHMARK_LOG=${CUR_DIR}/../../../../benchmark_log

        ## 获取平台类型，配置最大batch_size
        # cur_platform=""
        # get_platform cur_platform
        # mbs_name=max_batch_size_${cur_platform}
        ## 目前DLRM仅支持300系列
        batch_size=${batch_size}

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi
}

