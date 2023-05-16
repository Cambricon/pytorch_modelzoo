#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

ssd_vgg16_base_params () {
    device="mlu"
    dist_backend="cncl"

    batch_size="32"
    iters="2"
    eval_iters="2"
    lr="1e-3"
    precision="fp32"
    ddp="False" 
    card_num=1
    distributed="--device_id 0"
    evaluate="False";

    
    benchmark_mode="True"
    max_batch_size_MLU290="32"
    max_batch_size_MLU370="18"
    max_batch_size_MLU590_M9="256"
    max_batch_size_MLU590_M9U="256"
    max_batch_size_MLU590_H8="256"
    max_batch_size_MLU370_ECC="18"
    max_batch_size_V100="32"
}

set_configs () {
    # 调用相应网络的base_params
    ssd_vgg16_base_params

    # 根据每个字段的功能, overide对应参数
    args=$1
    params_array=(${args//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)   ;;
            O[0-3]) precision=$var ;;
            amp)    precision="amp" ;;
            mlu)    ;;
            gpu)    device="GPU" ;
		    dist_backend="nccl";;
            ddp)    ddp="True" ;
	    	    distributed="--multiprocessing-distributed";
		    batch_size="16";
		    lr="5e-4";;
            ci)     benchmark_mode="False";
                    evaluate="True";
                    ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done


    ## 加载公用方法
    source ${CONFIG_DIR}/../../../../../tools/utils/common_utils.sh
    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then

        ## 获取benchmark_mode计数规则,配置迭代数
        iters=-1
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

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi

    if [[ $ddp == "True" ]]; then
	get_visible_cards card_num
    fi

    if [[ $card_num -le 0 ]]; then
	echo "Invalid card number ${card_num}!!!"
	exit 1
    fi

    batch_size=`expr ${card_num} \* ${batch_size}`
}

