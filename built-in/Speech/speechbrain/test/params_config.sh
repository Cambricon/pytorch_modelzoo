#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)


ecapa_base_params () {
    device="mlu"
    precision="fp32"
    ddp="False"
    nproc_per_node=4
    train_params=""  # Train params is used to run checkin-mode(CI)
    benchmark_mode="True"
}

set_configs () {

    params=$1

    # 调用相应网络的base_params
    ecapa_base_params

    params_array=(${params//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
	    fp32)   ;;
	    O[0-3]) echo "net with CNMIX is not supported";
		    exit ;;
            amp)    precision="pyamp" ;
                    ;;
            mlu)    device="mlu:0";  # net use mlu:0 as default, and no need to specify it.
		            backend="cncl";
		            ;;
            gpu)    device="cuda:0" ;
		            backend="nccl";
		            ;;
            ddp)    ddp="True" ;
                    ;;
            ci)     train_params="--one_epoch --train_batches 5 --valid_batches 5 --eval_batches 5";
                    ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done
    source ${CONFIG_DIR}/../../../../tools/utils/common_utils.sh
    get_visible_cards nproc_per_node
    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then
        if [[ ${precision} == "pyamp" ]]; then
            train_params="--auto_mix_prec --one_epoch --train_batches 5 --valid_batches 5 --eval_batches 5"
        else
            train_params="--one_epoch --train_batches 60 --valid_batches 1 --eval_batches 1"
        fi

        ## 获取平台类型，配置最大batch_size
        cur_platform=""
        get_platform cur_platform
        # TODO(tyr): max batch size training is not tested at yet.
        mbs_name=max_batch_size_${cur_platform}

        cur_ecc_status=""
        get_ecc_status cur_ecc_status
        if [[ ${cur_ecc_status} == "ON" ]]; then
            mbs_name=max_batch_size_${cur_platform}_ECC
        fi
        batch_size=${!mbs_name}
        #nproc_per_node=${DEVICE_COUNT}

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi
}
