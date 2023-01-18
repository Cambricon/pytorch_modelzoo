#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

base_params () {
    device="mlu"
    batch_size="64"
    precision="fp32"
    ddp="False"
    train_iterations='1'
    evaluation=''

    benchmark_mode="True"
    num_workers="7"
    DEVICE_COUNT=1

    max_batch_size_MLU290="256"
    max_batch_size_MLU370="480"
    max_batch_size_MLU590="480"
    max_batch_size_MLU370_ECC="384"
    max_batch_size_V100="256"
}

set_configs () {
    params=$1

    base_params

    # 根据每个字段的功能, overide对应参数
    params_array=(${params//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)   ;;
            O[0-3]) precision=$var ;;
            amp)    precision="pyamp" ;;
            mlu)    ;;
            gpu)    device="gpu" ;;
            ddp)    ddp="True" ;;
            dummy_test)    dummy_test="True" ;;
            ci)     benchmark_mode=False;
                    train_iterations=2;
                    evaluation="--eval_iterations 2";
                    ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    if [[ $benchmark_mode == "False" ]]; then
        if [ -z ${MLU_VISIBLE_DEVICES} ]; then
            export MLU_VISIBLE_DEVICES=0
        fi
    fi

    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then
        ## 加载公用方法
        source ${CONFIG_DIR}/../../../../../tools/utils/common_utils.sh

        ## 获取benchmark_mode计数规则,配置迭代数
        perf_iters_rule train_iterations

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

        ## 检查多卡时是否设置VISIBLE_DEVICES环境变量
        visible_cards=-1
        get_visible_cards visible_cards
        DEVICE_COUNT=${visible_cards}
        if [[ $ddp == "True" ]]; then
            if [ $visible_cards -eq -1 ]; then
                echo "Please set env MLU_VISIBLE_DEVICES before running multicards."
                exit 1
            fi
        fi

        ## set num_workers for different platforms and cardnums
        num_workers="32"
        if [[ ${visible_cards} -eq 8 ]]; then
            num_workers="12"
        fi
        if [[ ${cur_platform} == "MLU370" ]]; then
            if [[ ${visible_cards} -eq 16 ]]; then
                num_workers="7"
            elif [[ ${visible_cards} -eq 4 ]]; then
                num_workers="16"
            fi
        fi

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi
}

