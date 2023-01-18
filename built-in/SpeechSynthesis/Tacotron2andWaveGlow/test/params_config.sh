#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

Tacotron2_base_params () {
    net="Tacotron2"

    use_mlu="True"
    batch_size="48"
    precision="fp32"
    lr="1e-3"
    output="./output/"
    weight_decay="1e-6"
    grad_clip_thresh="1.0"
    log_file="nvlog.json"
    anneal_factor="0.1"
    seed="123456"
    epochs="1501"
    iters="-1"
    anneal_steps="500 1000 1500"
    anneal_factor="0.1"
    cudnn_deterministic="False"

    ddp="False"
    num_cards="4"
    benchmark_mode="True"

    # max_batch_size_MLU290="128"
    max_batch_size_MLU370="48"
    max_batch_size_MLU590="48"
    max_batch_size_MLU370_ECC="48"
    max_batch_size_V100="48"

    #resume="${PYTORCH_TRAIN_CHECKPOINT}/TTS/checkpoint_Tacotron2_1500.pt"
}

WaveGlow_base_params () {
    net="WaveGlow"

    use_mlu="True"
    batch_size="4"
    precision="fp32"
    lr="1e-4"
    output="./output/"
    weight_decay="0"
    grad_clip_thresh_amp="65504.0"
    grad_clip_thresh="3.4028234663852886e+38"
    log_file="nvlog.json"
    seed="123456"
    epochs="1001"
    iters="-1"
    segment_length="8000"
    cudnn_deterministic="False"

    ddp="False"
    num_cards="4"
    benchmark_mode="True"

    # max_batch_size_MLU290="4"
    max_batch_size_MLU370="4"
    max_batch_size_MLU590="4"
    max_batch_size_MLU370_ECC="4"
    max_batch_size_V100="4"

    #resume="${PYTORCH_TRAIN_CHECKPOINT}/TTS/checkpoint_WaveGlow_1000.pt"
}
set_configs () {

    # 根据每个字段的功能, overide对应参数
    args=$1

    net=${args%%-*}
    params=${args#*-}

    ${net}_base_params

    params_array=(${params//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)   ;;
            O[0-3]) precision=$var ;;
            amp)    precision="pyamp" ;;
            mlu)    ;;
            gpu)    use_mlu="False";
                        cudnn_deterministic="True" ;;
            ddp)    ddp="True" ;;
            ci)     benchmark_mode="False";
                    iters="2";
                    ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then
        ## 加载公用方法
        source ${CONFIG_DIR}/../../../../tools/utils/common_utils.sh

        ## 获取benchmark_mode计数规则,配置迭代数
        iters=-1
        perf_iters_rule iters

        ## 检查MLU数量，获取最大卡片数
        num_cards=-1
        get_visible_cards num_cards

        if [[ $ddp == "True" ]]; then
            if [ $num_cards -eq -1 ]; then
                # echo "Please set env MLU_VISIBLE_DEVICES before running multicards."
                exit 1
            fi
        fi


        ## 设置benchmark_mode log路径
         #export BENCHMARK_LOG=${CUR_DIR}/../../benchmark_log

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
        pushd ${CONFIG_DIR}/../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi
}
