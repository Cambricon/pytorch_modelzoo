#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

transformer_base_params () {
    device="MLU"

    batch_size="512"
    num_epochs=10
    precision="fp32"
    num_workers="12"
    ddp="False"
    nproc_per_node=4
    iters=10
    evaluate="False";
    resume="False"

    benchmark_mode="True"
    max_batch_size_MLU290="32"
    max_batch_size_MLU370="512"
    max_batch_size_MLU590_M9="1280"
    max_batch_size_MLU590_M9U="1280"
    max_batch_size_MLU590_H8="1280"
    max_batch_size_MLU370_ECC="512"
    max_batch_size_V100="512"

}

set_configs () {
    # 调用相应网络的base_params
    transformer_base_params

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
            gpu)    device="GPU" ;;
            ddp)    ddp="True" ;;
            ci)     benchmark_mode=False;
                    iters=2;
                    evaluate="True";
                    resume="True";
                    ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    if [[ ${device} == "MLU" ]]; then
        DEVICE_COUNT=`echo $MLU_VISIBLE_DEVICES | awk -F, '{print NF}'`
    else
        DEVICE_COUNT=`echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}'`
    fi
    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then
        ## 加载公用方法
        source ${CONFIG_DIR}/../../../../tools/utils/common_utils.sh

        ## 获取benchmark_mode计数规则,配置迭代数
        iters=-1
        perf_iters_rule iters

        ## 获取平台类型，配置最大batch_size
        cur_platform=""
        get_platform_with_flag_name cur_platform
        mbs_name=max_batch_size_${cur_platform}

        cur_ecc_status=""
        get_ecc_status cur_ecc_status
        if [[ ${cur_ecc_status} == "ON" ]]; then
            mbs_name=max_batch_size_${cur_platform}_ECC
        fi
        batch_size=${!mbs_name}

        if [[ ${DEVICE_COUNT} -eq 16 || ${DEVICE_COUNT} -eq 8  ]]; then
            num_workers="7"
        fi
        nproc_per_node=${DEVICE_COUNT}
        num_epochs=1
        resume=""

        ## for bs=512, 获取benchmark_mode计数规则,配置迭代数
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
            iters=$total_iters
            export MLU_ADAPTIVE_STRATEGY_COUNT=$cutdown_iters
        fi

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi
}

