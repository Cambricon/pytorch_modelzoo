#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

bert_msra_base_params () {
    visible_cards="1"
    iters="1000"
    eval_iters="-1"
    lr="0.00003"
    batch_size="16"
    epoch="1"
    mode="train-eval"
    hvd_cards="-1"
    opt_level="-1"
    precision="fp32"
    device="mlu"

    benchmark_mode="True"
    max_batch_size_MLU290="16"
    max_batch_size_MLU370="24"
    max_batch_size_MLU590="24"
    max_batch_size_MLU370_ECC="16"
    max_batch_size_V100="32"
    ddp="False"

}

set_configs () {
    # 调用相应网络的base_params
    bert_msra_base_params
    ## 加载公用方法
    source ${CONFIG_DIR}/../../../../tools/utils/common_utils.sh

    # 根据每个字段的功能, overide对应参数
    args=$1
    params_array=(${args//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)   ;;
            O[0-3]) opt_level=$var ;
                    precision="cnmix"
                    ;;
            amp)    precision="pyamp" ;;
            mlu)    ;;
            gpu)    device="gpu" ;
                    opt_level="-1";
                    precision="fp32"
                    ;;
            ddp)    ddp="True";
		    get_visible_cards visible_cards
                    ;;
            ci)     benchmark_mode=False;
                    iters=2;
                    eval_iters="2"
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
        mbs_name=max_batch_size_${cur_platform}

        cur_ecc_status=""
        get_ecc_status cur_ecc_status
        if [[ ${cur_ecc_status} == "ON" ]]; then
            mbs_name=max_batch_size_${cur_platform}_ECC
        fi
        batch_size=${!mbs_name}

        ## 检查多卡时是否设置VISIBLE_DEVICES环境变量
        if [[ $ddp == "True"  ]]; then
            get_visible_cards visible_cards
            if [ $visible_cards -eq -1  ]; then
                echo "Please set env MLU_VISIBLE_DEVICES before running multicards."
                exit 1
            fi
        fi

        ## 设置benchmark时模型仅跑推理
        mode="train"

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi
}

