#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

crnn_base_params () {
    device="mlu"

    batch_size="64"
    lr="0.0001"
    beta="0.9"
    epoch=1
    precision="fp32"
    num_workers="4"
    ddp="False"

    benchmark_mode="True"
    display_interval=1
    max_batch_size_MLU290="16"
    max_batch_size_MLU370="96"
    max_batch_size_MLU590_M9="2048"
    max_batch_size_MLU590_M9U="2048"
    max_batch_size_MLU590_H8="1280"
    max_batch_size_MLU370_ECC="96"
    max_batch_size_V100="16"

    data_path="${PYTORCH_TRAIN_DATASET}/Synth90k/"
    resume="${PYTORCH_TRAIN_CHECKPOINT}/crnn/checkpoints_fp/netCRNN_14.pth"
}

set_configs () {
    # 调用相应网络的base_params
    crnn_base_params

    # 根据每个字段的功能, overide对应参数
    params=$1
    params_array=(${params//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)   ;;
            O[0-3]) precision=$var ;;
            mlu)    ;;
            gpu)    device="gpu" ;;
            ddp)    ddp="True";
                    batch_size=16;
                    ;;
            ci_train)  benchmark_mode=False;
                       iters=2;
                       ;;
            ci_eval)   benchmark_mode=False;
                       iters=2;
                       evaluate="True";
                       ;;
            dummy_test) dummy_test="True" ;;
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

        num_epochs=1
        resume=""
        visible_cards=-1
        get_visible_cards visible_cards
        ## 检查多卡时是否设置VISIBLE_DEVICES环境变量
        if [[ $ddp == "True" ]]; then
            if [ $visible_cards -eq -1 ]; then
                # echo "Please set env MLU_VISIBLE_DEVICES before running multicards."
                exit 1
            fi
        fi

        ## set num_workers for different platforms and cardnums
        num_workers="32"
        if [[ ${visible_cards} -eq 8 ]]; then
            num_workers="4"
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

