#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

swin_transformer_base_params () {
    model="swin_tiny_patch4_window7_224"
    batch_size="128"
    iters="200"
    epochs="1"
    opt="adamw"
    opt_eps="1.0e-08"
    opt_betas="0.9 0.999"
    momentum="0.9"
    lr="0.001"
    min_lr="1.0e-05"
    warmup_epochs="20"
    warmup_lr="1.0e-6"
    weight_decay="0.05"
    decay_rate="0.1"
    decay_epochs="30"
    clip_grad="5.0"
    device="mlu"
    color_jitter="0.4"
    cutmix="1.0"
    mixup="0.8"
    mixup_mode="batch"
    mixup_prob="1.0"
    recount="1"
    cooldown_epoch="0"
    remode="pixel"
    train_interpolation="bicubic"
    seed="0"
    drop_path="0.2"
    reprob="0.25"

    device_number="1"
    benchmark_mode="True"
    precision="fp32"
    ddp="False"


    max_batch_size_MLU290="240"
    max_batch_size_MLU370="136"
    max_batch_size_MLU370_ECC="136"
    max_batch_size_MLU370_AMP="160"
    max_batch_size_MLU590="256"
    max_batch_size_MLU590_AMP="260"
    max_batch_size_V100="128"

    dataset=$IMAGENET_TRAIN_DATASET
    output="./SwinTransformer/tmp"
}

set_configs () {
    params=$1

    # 调用相应网络的base_params
    swin_transformer_base_params
    ## 加载公用方法
    source ${CONFIG_DIR}/../../../../../tools/utils/common_utils.sh

    # 据每个字段的功能, overide对应参数
    params_array=(${params//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)   ;;
            amp)    precision="pyamp" ;;
            mlu)    ;;
            gpu)    device="gpu" ;;
            ddp)    ddp="True";
		    get_visible_cards device_number
		    ;;
            ci)  benchmark_mode=False;
                 iters=2;
                 eval_iters=2;
                 evaluate="True"
                ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then
        ## 获取benchmark_mode计数规则,配置迭代数
        iters=-1
        perf_iters_rule iters

        ## 设置benchmark_mode log路径
        export BENCHMARK_LOG=${CUR_DIR}/../../../benchmark_log

        ## 获取平台类型，配置最大batch_size
        cur_platform=""
        get_platform cur_platform
        mbs_name=max_batch_size_${cur_platform}

        cur_ecc_status=""
        get_ecc_status cur_ecc_status
        if [[ ${cur_ecc_status} == "ON" ]]; then
            mbs_name=max_batch_size_${cur_platform}_ECC
        fi

        if [[ $precision =~ "pyamp" ]]; then
           mbs_name=max_batch_size_${cur_platform}_AMP
        fi

        batch_size=${!mbs_name}

        ## 检查多卡时是否设置VISIBLE_DEVICES环境变量
        if [[ $ddp == "True" ]]; then
            visible_cards=-1
            get_visible_cards visible_cards
            if [ $visible_cards -eq -1 ]; then
                echo "Please set env MLU_VISIBLE_DEVICES before running multicards."
                exit 1
            fi
            device_number=${visible_cards}
        fi

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;

    fi
}

