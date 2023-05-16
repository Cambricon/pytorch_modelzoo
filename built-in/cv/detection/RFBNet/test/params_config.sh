#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

base_params () {
    device="mlu"

    num_data="16512"
    batch_size="8"
    max_rounds=1

    lr="0.001"
    precision="fp32"
    ddp="False"
    num_workers="12"

    world_size=1
    node_rank=0
    evaluate="False"

    benchmark_mode="True"
    max_batch_size_MLU290="32"
    max_batch_size_MLU370="64"
    max_batch_size_MLU590_M9="184"
    max_batch_size_MLU590_M9U="184"
    max_batch_size_MLU590_H8="184"
    max_batch_size_MLU370_ECC="64"
    max_batch_size_V100="64"

    resume_opt="--resume --resume_epoch 150"
    mode_str="benchmark"
    output="${CONFIG_DIR}/rfbnet_mlu"
}

set_configs () {
    params=$1

    # 调用网络的base_params
    base_params

    # 根据每个字段的功能, overide对应参数
    params_array=(${params//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)   ;;
            O[0-3]) precision=$var ;;
            amp)    precision="pyamp" ;;
            mlu)    ;;
            gpu)    device="cuda";
                    output="${CONFIG_DIR}/rfbnet_gpu" ;;
            ddp)    ddp="True";
                    max_rounds=1;;
            ci)     benchmark_mode=False;
                    mode_str="precheckin"
                    resume_opt="--resume --resume_epoch 150 --unit_in_iters"
                    max_rounds=2
                    card_num=1;
                    evaluate="True" ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then
        ## iters isn't use,  only to export MLU_ADAPTIVE_STRATEGY_COUNT=100
        iters=-1

        ## 加载公用方法
        source ${CONFIG_DIR}/../../../../../tools/utils/common_utils.sh
        perf_iters_rule iters

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

        card_num=1
           ## 检查多卡时是否设置VISIBLE_DEVICES环境变量
        if [[ $ddp == "True" ]]; then
            visible_cards=-1
            distributed_flag='--distributed'
            get_visible_cards visible_cards
            if [ $visible_cards -eq -1 ]; then
                # echo "Please set env MLU_VISIBLE_DEVICES before running multicards."
                exit 1
            fi
            card_num=$visible_cards
            batch_size=$(($batch_size*$card_num))
	    if [[ $card_num -eq 16 ]]; then
		    num_workers="7"
	    fi
	    num_workers=$(($num_workers*$card_num))
        fi
        
	num_iters=$[ $num_data / $batch_size ]
        if [[ $num_iters -lt $iters ]]; then
            iters=$num_iters
	    iters_count=$[ $num_iters / 3 ]
            export MLU_ADAPTIVE_STRATEGY_COUNT=$iters_count
        fi

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi
}
