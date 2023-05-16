#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

base_params () {
    device_param="mlu"

    train_iters=-1
    eval_iters=-1
    num_train_epochs=1
    batch_size="16"
    precision="fp32"
    num_workers="8"
    nproc_per_node=4
    eval_iters="-1"
    evaluate="False";

    model_type="bert"
    lr="3e-5"
    max_seq_length="384"
    doc_stride="128"
    output_dir="bert_base_cased_from_scratch/"
    train_file="${SQUAD_DIR}/train-v1.1.json"
    predict_file="${SQUAD_DIR}/dev-v1.1.json"

    use_launch=""
    nnodes=1

    benchmark_mode="True"
    max_batch_size_MLU290="16"
    max_batch_size_MLU370="32"
    max_batch_size_MLU590_M9="160"
    max_batch_size_MLU590_M9U="160"
    max_batch_size_MLU590_H8="96"
    max_batch_size_MLU370_AMP="32"
    max_batch_size_MLU590_M9_AMP="160"
    max_batch_size_MLU590_M9U_AMP="160"
    max_batch_size_MLU590_H8_AMP="96"
    max_batch_size_MLU370_ECC="32"
    max_batch_size_V100="16"

}

set_configs () {
    # 调用相应网络的base_params
    base_params

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
            gpu)    device_param="gpu" ;;
            ddp)    ddp="True"
                    output_dir="bert_base_cased_ddp_from_scratch/" ;;
            ci)     benchmark_mode=False;
                    train_iters=2;
                    eval_iters="2"
                    num_train_epochs="1";
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
        perf_iters_rule train_iters

        ## 设置benchmark_mode log路径
        #export BENCHMARK_LOG=${CUR_DIR}/../../../../benchmark_log

        ## 获取平台类型，配置最大batch_size
        cur_platform=""
        get_platform_with_flag_name cur_platform
        mbs_name=max_batch_size_${cur_platform}

        get_ecc_status cur_ecc_status
        if [[ ${cur_ecc_status} == "ON" ]]; then
            mbs_name=max_batch_size_${cur_platform}_ECC
        fi

        if [[ $precision == "pyamp" ]]; then
            mbs_name=max_batch_size_${cur_platform}_AMP
        fi

        batch_size=${!mbs_name}

        ## 检查多卡时是否设置VISIBLE_DEVICES环境变量
        visible_cards=-1
        if [[ $ddp == "True" ]]; then
            get_visible_cards visible_cards
            if [ $visible_cards -eq -1 ]; then
                # echo "Please set env MLU_VISIBLE_DEVICES before running multicards."
                exit 1
            fi
        fi

        if [[ $ddp == "True" ]]; then
            use_launch="-m torch.distributed.launch --nproc_per_node=${visible_cards} --nnodes=${nnodes}"
        fi

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi
}

