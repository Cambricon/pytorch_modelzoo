#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

base_params () {
    precision="fp32"
    ddp="False"
    device="mlu"
    dist_backend="cncl"
    DEVICE_COUNT=1
    seed=2
    max_epochs=1
    learning_rate=0.0005
    batch_size=2
    lr_decay=0.5
    data="$PYTORCH_TRAIN_DATASET/CityScapes"
    width=1024
    height=512
    nproc_per_node=4
    benchmark_mode="True"
    debug_mode="False"
    
    max_batch_size_MLU370="16"
    max_batch_size_MLU590="16"
    max_batch_size_MLU370_ECC="16"
    max_batch_size_V100="8"
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
       ci_train)    benchmark_mode=False;
	            debug_mode=True;
                    train_steps=2;
                    ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    if [[ $benchmark_mode == "False" ]]; then
        export WORLD_SIZE=1
        export RANK=0
    fi

    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then
        ## 加载公用方法
        source ${CONFIG_DIR}/../../../../../tools/utils/common_utils.sh

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

        visible_cards=1
        use_launch="-m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 "
        if [[ $ddp == "True" ]]; then
            get_visible_cards visible_cards
        fi
        ## 检查多卡时是否设置VISIBLE_DEVICES环境变量
        if [[ $ddp == "True" ]]; then
            use_launch="-m torch.distributed.launch --nproc_per_node=${visible_cards} --nnodes=1 --node_rank=0 "
            if [ $visible_cards -eq -1 ]; then
                echo "Please set env MLU_VISIBLE_DEVICES before running multicards."
                exit 1
            fi
        fi

        # 当每个epoch的step不足20时，需要修改cutdown_iters和total_iters
        if [[ $cur_platform == "MLU370" || $cur_platform == "MLU590" ]]; then
            total_iters=`expr 2975 / $visible_cards / $batch_size`
            cutdown_iters=20
            if [[ ${visible_cards} -lt 4 ]]; then
                perf_iters_rule total_iters
            elif [[ ${visible_cards} -le 8 ]]; then
                cutdown_iters=10
            elif [[ ${visible_cards} -gt 8 ]]; then
                cutdown_iters=5
            fi
        fi
        export MLU_ADAPTIVE_STRATEGY_COUNT=$cutdown_iters
        train_steps=$total_iters

        ## 检查多卡时是否设置VISIBLE_DEVICES环境变量
        if [[ $ddp == "True" ]]; then
            if [ $visible_cards -eq -1 ]; then
                echo "Please set env MLU_VISIBLE_DEVICES before running multicards."
                exit 1
            fi
        fi

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi
}
