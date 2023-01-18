#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)
echo CONFIG_DIR
base_params () {
    precision="fp32"
    ddp="False"
    DEVICE_COUNT='1'

    seed=0

    max_epochs=1
    quality_threshold="0.908"
    start_eval_at=1000
    evaluate_every=1
    learning_rate="3.2"
    lr_warmup_epochs=200
    dataset_dir="$PYTORCH_TRAIN_DATASET/KiTS19/pre_data_dir"
    batch_size=2
    gradient_accumulation_steps=1
    train_steps=-1
    eval_steps=-1

    use_launch=""
    nproc_per_node="4"

    benchmark_mode="True"

    max_batch_size_MLU290="2"
    max_batch_size_MLU370="2"
    max_batch_size_MLU590="2"
    max_batch_size_MLU370_ECC="2"
    max_batch_size_V100="2"
}

set_configs () {
    params=$1

    base_params

    # 根据每个字段的功能, overide对应参数
    params_array=(${params//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)   ;;
            O[0-3]) precision=$var ;;
            amp)    precision="amp" ;;
            mlu)    ;;
            gpu)    device_param="gpu" ;;
            ddp)    ddp="True"
                    ;;
            ci)     benchmark_mode=False;
                    train_steps=2;
                    eval_steps=2;
                    ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    if [[ $benchmark_mode == "False" ]]; then
        if [ -z ${MLU_VISIBLE_DEVICES} ]; then
            export WORLD_SIZE=1
        fi
    fi

    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then
        ## 加载公用方法
        source ${CONFIG_DIR}/../../../../tools/utils/common_utils.sh

        ## 获取benchmark_mode计数规则,配置迭代数
        perf_iters_rule train_steps

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

        visible_cards=-1
        get_visible_cards visible_cards
        ## 检查多卡时是否设置VISIBLE_DEVICES环境变量
        if [[ $ddp == "True" ]]; then
            use_launch="-m torch.distributed.launch --nproc_per_node=${visible_cards} "
            if [ $visible_cards -eq -1 ]; then
                echo "Please set env MLU_VISIBLE_DEVICES before running multicards."
                exit 1
            fi
        fi

        ## 检查多卡时是否设置VISIBLE_DEVICES环境变量
        if [[ $ddp == "True" ]]; then
            if [ $visible_cards -eq -1 ]; then
                echo "Please set env MLU_VISIBLE_DEVICES before running multicards."
                exit 1
            fi
            ## 168 images per epoch, need set small train_steps if DDP enabled
            if [ $visible_cards -ge 2 ]; then
                export MLU_ADAPTIVE_STRATEGY_COUNT=2
                train_steps=4
            fi
        fi

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi
}

