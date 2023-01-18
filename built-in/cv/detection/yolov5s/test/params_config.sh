#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

yolov5s_base_params () {
    net="yolov5s"
    device="mlu"

    batch_size="16"
    seed="66"
    epochs="152"
    precision="fp32"
    num_workers="16"
    ddp="False"

    benchmark_mode="True"
    evaluate="False"
    max_batch_size_MLU290="16"
    max_batch_size_MLU370="80"
    max_batch_size_MLU590="80"
    max_batch_size_MLU370_ECC="64"
    max_batch_size_V100="16"

    yolo_path=$CONFIG_DIR/../../
    data=data/coco.yaml
    cfg=models/yolov5s.yaml
    origin_weight=$PYTORCH_TRAIN_CHECKPOINT/yolov5/origin_epoch_150.pth
    ema_weight=$PYTORCH_TRAIN_CHECKPOINT/yolov5/epoch_150.pth
    resume="True"
}

set_configs () {
    params=$1

    # 调用相应网络的base_params
    yolov5s_base_params

    # 根据每个字段的功能, overide对应参数
    params_array=(${params//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)   ;;
            O[0-3]) precision=$var ;;
            amp)    precision="pyamp" ;;
            mlu)    ;;
            gpu)    device="gpu" ;;
            ddp)    ddp="True" ;;
            ci_train)  benchmark_mode=False;
                       iters=2;
                       ;;
            ci_eval)   benchmark_mode=False;
                       eval_iters=2;
                       evaluate="True"
                       ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then
        ## 加载公用方法
        source ${CONFIG_DIR}/../../../../../tools/utils/common_utils.sh

        ## 获取benchmark_mode计数规则,配置迭代数
        iters=-1
        perf_iters_rule iters

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

        ## set num_workers for different platforms and cardnums
        visible_cards=-1
        get_visible_cards visible_cards
        num_workers="8"
        if [[ ${visible_cards} -gt 0 && ${ddp} == "True" ]]; then
            num_workers=`expr ${num_workers} \* ${visible_cards}`
        fi

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;

    fi

    # 配置推理时的参数
    if [[ ${evaluate} == "True" ]]; then
        weights=weights/${device}/checkpoint.pth.tar
    fi
}
