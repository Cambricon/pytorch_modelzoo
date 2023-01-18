#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

inception_v3_base_params () {
    model="inception_v3"
    batch_size="64"
    iters="200"
    epochs="1"
    sched="cosine"
    opt="sgd"
    opt_eps=".001"
    lr="0.045"
    warmup_lr="0.0001"
    weight_decay="4e-5"
    decay_rate="0.94"
    decay_epochs="4"
    device="mlu"
    cooldown_epoch="0"
    remode="const"
    reprob="0."
    drop="0.0"
    aa="rand-m9-mstd0.5"
    num_workers=64

    visible_cards=1
    benchmark_mode="True"
    precision="fp32"
    ddp="False"

    benchmark_mode="True"
    max_batch_size_MLU370="104"
    max_batch_size_MLU370_AMP="180"
    max_batch_size_MLU590_M9="180"
    max_batch_size_MLU590_H8="180"
    max_batch_size_MLU590_M9_AMP="180"
    max_batch_size_MLU590_H8_AMP="180"
    max_batch_size_MLU370_ECC="104"
    max_batch_size_A100="104"

    dataset=$IMAGENET_TRAIN_DATASET
    output="./output/inceptionv3_train"
}


inception_v4_base_params () {
    model="inception_v4"
    batch_size="104"
    iters="200"
    epochs="1"
    sched="step"
    opt="rmsproptf"
    opt_eps=".001"
    lr="0.0224"
    warmup_lr="3.5e-7"
    weight_decay="1e-5"
    decay_rate=".973"
    decay_epochs="2.4"
    device="mlu"
    cooldown_epoch="0"
    remode="pixel"
    reprob="0.2"
    drop="0.2"
    aa="rand-m9-mstd0.5"
    lr_noise="0.42 0.9"
    num_workers=64

    visible_cards=1
    benchmark_mode="True"
    precision="fp32"
    ddp="False"

    benchmark_mode="True"
    max_batch_size_MLU370="104"
    max_batch_size_MLU370_AMP="180"
    max_batch_size_MLU590_M9="180"
    max_batch_size_MLU590_H8="180"
    max_batch_size_MLU590_M9_AMP="180"
    max_batch_size_MLU590_H8_AMP="180"
    max_batch_size_MLU370_ECC="104"
    max_batch_size_A100="104"

    dataset=$IMAGENET_TRAIN_DATASET
    output="./output/inceptionv4_train"
}

set_configs () {
    args=$1

    # 获取网络和参数字段
    net=${args%%-*}
    params=${args#*-}

    # 调用相应网络的base_params
    ${net}_base_params

    ## 加载公用方法
    source ${CONFIG_DIR}/../../../../../tools/utils/common_utils.sh

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
            ddp)    ddp="True" ;
                    get_visible_cards visible_cards;;
            ci)     benchmark_mode=False;
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
        export BENCHMARK_LOG=${CUR_DIR}/../../../../benchmark_log

        ## 获取平台类型，配置最大batch_size
        cur_platform=""
        get_platform cur_platform
        if [[ $precision =~ 'amp' ]];then
            mbs_name=max_batch_size_${cur_platform}_AMP
        else
            mbs_name=max_batch_size_${cur_platform}
        fi

        cur_ecc_status=""
        get_ecc_status cur_ecc_status
        if [[ ${cur_ecc_status} == "ON" ]]; then
            mbs_name=max_batch_size_${cur_platform}_ECC
        fi

        if [[ $precision == "amp" ]]; then
            mbs_name=max_batch_size_${cur_platform}_AMP
        fi

        batch_size=${!mbs_name}

        ## set num_workers
        num_workers="12"

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi

    if [[ $ddp == "True" ]];then
        if [[ ${visible_cards} -eq 0 || ${visible_cards} -eq -1 ]]; then
            exit 1
        fi
        export MASTER_ADDR=localhost
        export MASTER_PORT=12346
    fi
}

