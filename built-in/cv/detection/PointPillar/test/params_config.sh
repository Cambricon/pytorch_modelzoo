#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

base_params () {
    device="mlu"

    batch_size="2"
    train_iters="2"
    eval_iters="2"
    lr="0.003"
    precision="fp32"
    launcher="none"
    card_num=1
    total_epochs="11"
    cfg_file="cfgs/nuscenes_models/cbgs_pp_multihead.yaml"

    benchmark_mode="True"
    max_batch_size_MLU370="8"
    max_batch_size_MLU590_M9="48"
    max_batch_size_MLU590_M9U="48"
    max_batch_size_MLU590_H8="40"
    max_batch_size_MLU370_ECC="8"
    max_batch_size_V100="8"

    ckpt="${PYTORCH_TRAIN_CHECKPOINT}/PointPillar/checkpoint_epoch_10.pth"
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
            amp)    precision="amp" ;;
            mlu)    ;;
            gpu)    device="gpu" ;;
            ddp)    launcher="pytorch" ;;
            ci_train)     benchmark_mode=False ;;
            ci_eval)     benchmark_mode=False;
                         evaluate="True" ;;
            *) echo "Unrecognized option: " $var; exit 1 ;;
        esac
    done

    ## 加载公用方法
    source ${CONFIG_DIR}/../../../../../tools/utils/common_utils.sh

    ## 多卡的话获取卡数
    if [[ $launcher == "pytorch" ]]; then
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

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi
}


