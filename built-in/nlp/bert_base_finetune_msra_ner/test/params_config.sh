#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

bert_msra_base_params () {
    device="mlu"

    num_data="40960"
    batch_size="30"
    run_epochs="1"
    precision="fp32"
    num_workers="8"
    ddp="False"
    nproc_per_node=4
    eval_iters="-1"
    evaluate="False";

    benchmark_mode="True"
    max_batch_size_MLU290="30"
    max_batch_size_MLU370="40"
    max_batch_size_MLU590_M9="256"
    max_batch_size_MLU590_M9U="256"
    max_batch_size_MLU590_H8="160"
    max_batch_size_MLU370_ECC="40"
    max_batch_size_V100="30"

}

set_configs () {
    # 调用相应网络的base_params
    bert_msra_base_params

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
                    eval_iters="2"
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

        device="mlu"
        ## 设置benchmark_mode log路径
        #export BENCHMARK_LOG=${CUR_DIR}/../../../../benchmark_log

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

        visible_cards=-1
        get_visible_cards visible_cards
        ## 检查多卡时是否设置VISIBLE_DEVICES环境变量
        if [[ $ddp == "True" ]]; then
            if [ $visible_cards -eq -1 ]; then
                # echo "Please set env MLU_VISIBLE_DEVICES before running multicards."
                exit 1
            fi
	    num_data_per_iter=$[ $batch_size * $visible_cards]
            num_iters=$[ $num_data / $num_data_per_iter ]
            if [[ $num_iters -lt $iters  ]]; then
                iters=$num_iters
                iters_count=$[ $num_iters / 3 ]
                export MLU_ADAPTIVE_STRATEGY_COUNT=$iters_count
            fi
        fi

        if [[ ${visible_cards} -eq 16 || ${visible_cards} -eq 8  ]]; then
            num_workers="7"
        fi
        nproc_per_node=${visible_cards}

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi
}

