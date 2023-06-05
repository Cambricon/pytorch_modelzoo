#!/bin/bash

CONFIG_DIR=$(
    cd $(dirname $0)
    pwd
)

mt5_base_params() {
    device="MLU"

    batch_size="4"
    num_epochs=1
    precision="fp32"
    ddp="False"
    nproc_per_node=4
    train_iters=-1
    valid_iters=-1
    eval_iters=-1
    evaluate="False"

    benchmark_mode="True"
    max_batch_size_MLU370="8"
    max_batch_size_MLU590="16"
    max_batch_size_MLU370_ECC="8"
    max_batch_size_V100="4"

}

set_configs() {
    # 调用相应网络的base_params
    mt5_base_params

    # 根据每个字段的功能, overide对应参数
    args=$1
    params_array=(${args//-/ })
    for var in ${params_array[@]}; do
        case "$var" in
        fp32) 
            max_batch_size_MLU370="6"
            max_batch_size_MLU370_ECC="6"
        ;;
        amp) precision="amp" ;;
        mlu) ;;
        gpu) device="GPU" ;;
        ddp) ddp="True" ;;
        ci)
            benchmark_mode=False
            train_iters=10
            valid_iters=10
            evaluate="True"
            ;;
        *)
            echo "Unrecognized option: " $var
            exit 1
            ;;
        esac
    done

    if [[ ${device} == "MLU" ]]; then
        DEVICE_COUNT=$(echo $MLU_VISIBLE_DEVICES | awk -F, '{print NF}')
    else
        DEVICE_COUNT=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')
    fi
    nproc_per_node=${DEVICE_COUNT}
    # 处理benchmark_mode所需的参数
    num_epochs=1
    ## 加载公用方法
    source ${CONFIG_DIR}/../../../../tools/utils/common_utils.sh
    ## 获取平台类型，配置最大batch_size
    cur_platform=""
    get_platform cur_platform
    mbs_name=max_batch_size_${cur_platform}

    cur_ecc_status=""
    get_ecc_status cur_ecc_status
    if [[ ${cur_ecc_status} == "ON" ]]; then
        mbs_name=max_batch_size_${cur_platform}_ECC
    fi
    batch_size=${!mbs_name}
    if [[ $benchmark_mode == "True" ]]; then
        ## 获取benchmark_mode计数规则,配置迭代数
        train_iters=-1
        perf_iters_rule train_iters

        if [[ $cur_platform == "MLU370" && ${ddp} == "True" ]]; then
            total_iters=100
            cutdown_iters=50
            if [[ ${DEVICE_COUNT} -le 4 ]]; then
                total_iters=20
                cutdown_iters=10
            elif [[ ${DEVICE_COUNT} -le 8 ]]; then
                total_iters=10
                cutdown_iters=5
            elif [[ ${DEVICE_COUNT} -le 16 ]]; then
                total_iters=6
                cutdown_iters=2
            fi
            train_iters=$total_iters
            export MLU_ADAPTIVE_STRATEGY_COUNT=$cutdown_iters
        fi

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../tools/mlu_performance_check/
        ./check_mlu_perf.sh
        popd
    fi
}
