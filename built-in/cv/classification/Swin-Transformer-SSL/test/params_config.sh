#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

base_params () {
    device="mlu"
    batch_size="64"
    precision="fp32"
    ddp="False"
    visible_cards=1
    num_workers="8"
    iters=-1
    eval_iters=-1

    benchmark_mode="True"
    max_batch_size_MLU370="64"
    max_batch_size_MLU590="64"
    max_batch_size_MLU370_ECC="64"
    max_batch_size_V100="32"

    resume="True"

    YAML_CONFIG="configs/moby_swin_tiny.yaml"
    OUTPUT_DIR="./output"
    PRETRAINED_PTH=$IMAGENET_TRAIN_CHECKPOINT/swin_transformer_ssl/pretrain_ckpt_epoch_299.pth
    RESUME_PRETRAINED=$IMAGENET_TRAIN_CHECKPOINT/swin_transformer_ssl/pretrain_ckpt_epoch_200.pth
    RESUME_LINEAR=$IMAGENET_TRAIN_CHECKPOINT/swin_transformer_ssl/linear_ckpt_epoch_50.pth
}

set_configs () {
    # 调用相应网络的base_params
    base_params

    ## 加载公用方法
    source ${CONFIG_DIR}/../../../../../tools/utils/common_utils.sh

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
            ddp)    ddp="True" ;
                    get_visible_cards visible_cards;;
            ci)     benchmark_mode=False;
                    iters=2;
                    eval_iters=2;
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
        #export BENCHMARK_LOG=${CUR_DIR}/../../../../benchmark_log

        ## 获取平台类型，配置最大batch_size
        cur_platform=""
        get_platform cur_platform
        mbs_name=max_batch_size_${cur_platform}
        batch_size=${!mbs_name}

        ## set num_workers
        num_workers="12"

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi

    if [[ $ddp == "True" ]];then
        if [[ ${visible_cards} -eq 0 || ${visible_cards} -eq -1 ]]; then
            exit 1
        fi
        export MASTER_ADDR=localhost
        export MASTER_PORT=12345
    fi
}

