#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

deepspeech2_base_params () {
    device="mlu"
    num_workers="4"
    benchmark_mode="True"
    ddp="False"
    resume="True"
    iters=-1
    eval_iters=-1
    evaluate="False";
}

set_configs () {
    # 调用相应网络的base_params
    deepspeech2_base_params

    # 根据每个字段的功能, overide对应参数
    args=$1
    params_array=(${args//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)         ;;
            O[0-3])       echo "DeepSpeech2 do not supported CNMIX, please run precision fp32.";
                          exit 1;;
            amp)          echo "DeepSpeech2 do not supported AMP, please run precision fp32.";
                          exit 1;;
            mlu)          ;;
            gpu)          device="gpu" ;;
            ddp)          ddp="True" ;;
            ci)           benchmark_mode=False;
                          evaluate="True";
                          iters=2;
                          eval_iters=2;
                          ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    ## 加载公用方法
    source ${CONFIG_DIR}/../../../../tools/utils/common_utils.sh

    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then
        ## 获取benchmark_mode计数规则,配置迭代数
        perf_iters_rule iters

        ## 获取平台类型，配置最大batch_size
        cur_platform=""
        get_platform cur_platform
        mbs_name=max_batch_size_${cur_platform}
        batch_size=${!mbs_name}

        ## 设置num_workers
        if [[ $ddp == "True" ]]; then
            num_workers="12"
        fi
        if [[ ${device} == "mlu" ]]; then
            DEVICE_COUNT=`echo $MLU_VISIBLE_DEVICES | awk -F, '{print NF}'`
        else
            DEVICE_COUNT=`echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}'`
        fi
        if [[ ${DEVICE_COUNT} -eq 16 || ${DEVICE_COUNT} -eq 8 ]]; then
            num_workers="7"
        fi

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi

    ## 获取cards_num
    cards_num=-1
    if [[ $ddp == "True" ]]; then
        get_visible_cards cards_num
        echo "$cards_num ++++++++++++++++++++++++++++++++++++++++"
        if [ $cards_num -eq -1 ]; then
            echo "Please set env MLU_VISIBLE_DEVICES before running multicards."
            exit 1
        fi
    fi
}

