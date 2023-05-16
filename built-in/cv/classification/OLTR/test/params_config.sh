#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

bbn_base_params () {
    device="mlu"
    batch_size="128"
    precision="fp32"
    num_workers="32"
    ddp="False"

    benchmark_mode="True"
    max_batch_size_MLU290="128"
    max_batch_size_MLU370="184"
    max_batch_size_MLU590_M9="1024"
    max_batch_size_MLU590_M9U="1024"
    max_batch_size_MLU590_H8="840"
    max_batch_size_MLU370_ECC="184"
    max_batch_size_V100="128"

    resume="False"
    evaluate="False";
}

set_configs () {
    # 调用相应网络的base_params
    bbn_base_params

    # 根据每个字段的功能, overide对应参数
    args=$1
    params_array=(${args//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)   ;;
            O[0-3]) precision=$var ;;
            amp)    precision="pyamp" ;;
            mlu)    ;;
            gpu)    device="gpu" ;;
            ddp)    ddp="True" ;;
            ci)     benchmark_mode=False;
                    iters=2;
                    evaluate="True";
                    ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then
        ## 加载公用方法
        source ${CONFIG_DIR}/../../../../../tools/utils/common_utils.sh

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

        resume=""
        ## set num_workers for different platforms and cardnums
        visible_cards=-1
        get_visible_cards visible_cards
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

    # Todo : 以下修改未考虑GPU，后续统一添加
    if [[ $ddp == "True" ]];then
        if [ $visible_cards -eq -1 ];then
            exit 1
        fi
        runnable_cards=$MLU_VISIBLE_DEVICES
    else
        if [ $visible_cards -eq -1 ];then
            runnable_cards=0
        else
            runnable_cards=`echo $MLU_VISIBLE_DEVICES | awk -F "," '{print $1}'`
        fi
    fi

}

