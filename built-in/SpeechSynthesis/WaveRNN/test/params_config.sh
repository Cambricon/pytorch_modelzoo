#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

wavernn_base_params () {
    epochs='1'
    net="wavernn"
    device="mlu"
    lr="0.0001"
    seed="123456"
    batch_size="32"
    precision="fp32"
    num_workers="8"
    ddp="False"
    benchmark_mode="True"
    num_per_checkpoint="10"
    max_batch_size_MLU290="32"
    max_batch_size_MLU370="32"
    max_batch_size_MLU590_M9="1440"
    max_batch_size_MLU590_M9U="1440"
    max_batch_size_MLU590_H8="1024"
    max_batch_size_MLU590_M9_AMP="1440"
    max_batch_size_MLU590_M9U_AMP="1440"
    max_batch_size_MLU590_H8_AMP="1024"
    max_batch_size_V100="32"

    resume="False"
}

set_configs () {
    wavernn_base_params

    # 根据每个字段的功能, overide对应参数
    params=$1
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
        source ${CONFIG_DIR}/../../../../tools/utils/common_utils.sh

        ## 获取benchmark_mode计数规则,配置迭代数
        iters=-1
        perf_iters_rule iters

        ## 设置benchmark_mode log路径
        #export BENCHMARK_LOG=${CUR_DIR}/../../../../benchmark_log

        ## 获取平台类型，配置最大batch_size
        #cur_platform=""
        #get_platform cur_platform
        #mbs_name=max_batch_size_${cur_platform}
        #目前WaveRNN仅支持300系列
        batch_size=${batch_size}

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi
}

