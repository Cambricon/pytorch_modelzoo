#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

p3d_base_params () {
    device="mlu"
    dataset="$PYTORCH_TRAIN_DATASET/ucf101/"
    backend="cncl"
    device_number=1

    batch_size="16"
    lr="0.5e-3"
    dropout="0.9"
    seed="42"
    epochs="1"
    train_steps="-1"
    eval_steps="-1"
    precision="fp32"
    num_workers="4"
    ddp="False"
    print_freq="20"

    benchmark_mode="True"
    max_batch_size_MLU290="16"
    max_batch_size_MLU370="40"
    max_batch_size_MLU590="40"
    max_batch_size_MLU370_ECC="40"
    max_batch_size_V100="16"

    resume="${IMAGENET_TRAIN_CHECKPOINT}/p3d/p3d_ckp_10.pth.tar"
    log_dir="${CUR_DIR}/../p3d_one_card_log/"
}

set_configs () {
    args=$1

    # 调用相应网络的base_params
    p3d_base_params
     ## 加载公用方法
    source ${CONFIG_DIR}/../../../../../tools/utils/common_utils.sh

    # 根据每个字段的功能, overide对应参数
    params_array=(${args//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)   ;;
            O[0-3]) precision=$var ;;
            amp)    precision="amp" ;;
            mlu)    ;;
            gpu)    device="gpu";
                    backend="nccl" ;;
            ddp)    ddp="True";
                    get_visible_cards device_number ;;
            ci)     benchmark_mode=False;
                    train_steps="2";
                    eval_steps="2";
                    print_freq="1"
                    epochs=11;
                    resume_multi_device="True";
                    ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    if [[ ${device_number} -eq 0 || ${device_number} -eq -1 ]]; then
        exit
    fi

    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then
        ## 设置benchmark_mode log路径
        export BENCHMARK_LOG=${CUR_DIR}/../../../../benchmark_log
        log_dir="${CUR_DIR}/../p3d_${device_number}_card_log/"

        ## 获取平台类型，配置最大batch_size
        if [[ $device != "gpu" ]]; then
            cur_platform=""
            get_platform cur_platform
            mbs_name=max_batch_size_${cur_platform}

            cur_ecc_status=""
            get_ecc_status cur_ecc_status
            if [[ ${cur_ecc_status} == "ON" ]]; then
                mbs_name=max_batch_size_${cur_platform}_ECC
            fi
            batch_size=${!mbs_name}
        fi

        ## set num_workers for different platforms and cardnums
        num_workers="32"
        if [[ ${device_number} -eq 8 ]]; then
            num_workers="12"
        fi
        if [[ ${cur_platform} == "MLU370" ]]; then
            if [[ ${device_number} -eq 16 ]]; then
                num_workers="7"
            elif [[ ${device_number} -eq 4 ]]; then
                num_workers="16"
            fi
        fi

        resume=""
        ## 获取benchmark_mode计数规则,配置迭代数
        total_iters=40
        cutdown_iters=20
        if [[ $cur_platform == "MLU290" ]]; then
            total_iters=120
            cutdown_iters=40
            if [[ ${device_number} -eq 8 ]]; then
                total_iters=40
                cutdown_iters=20
            fi
        else
            if [[ ${device_number} -eq 8 ]]; then
                total_iters=20
                cutdown_iters=10
            elif [[ ${device_number} -eq 16 ]]; then
                total_iters=10
                cutdown_iters=5
            fi
        fi

        train_steps=$total_iters
        export MLU_ADAPTIVE_STRATEGY_COUNT=$cutdown_iters
        # 注意这里跑的总代数为min(train_steps, epochs)
        # 这里设置的epochs为任意大于train_steps对应epoch的大数
        epochs="10"
    fi
}

