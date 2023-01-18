#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

base_params () {
    device="mlu"

    batch_size="2"
    base_iters="0"
    train_iters="2"
    eval_iters="2"
    lr="0.0025"
    precision="fp32"
    ddp="False"
    card_num=4
    num_workers="4"
    evaluate="False";

    benchmark_mode="True"
    max_batch_size_MLU290="2"
    max_batch_size_MLU370="4"
    max_batch_size_MLU590="4"
    max_batch_size_MLU370_ECC="4"
    max_batch_size_V100="2"
}

set_configs () {
    params=$1

    # 调用网络的base_params
    base_params

    # 根据每个字段的功能, overide对应参数
    params_array=(${params//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)   ;;
            O[0-3]) precision=$var ;;
            mlu)    ;;
            gpu)    device="gpu" ;;
            ddp)    ddp="True";;
            ci)     benchmark_mode=True;
                    evaluate="True" ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    ## 加载公用方法
    source ${CONFIG_DIR}/../../../../../tools/utils/common_utils.sh

    if [[ $ddp == "True" ]]; then
        get_visible_cards card_num
    fi

    if [[ $card_num -le 0 ]]; then
        echo "Invalid card number ${card_num} !!!"
        exit 1
    fi

    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then

        ## 获取benchmark_mode计数规则,配置迭代数
        eval_iters=-1
        train_iters=-1
        perf_iters_rule train_iters
        train_iters=$(( base_iters + ${train_iters}))

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

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi

    batch_size=`expr ${card_num} \* ${batch_size}`
}
