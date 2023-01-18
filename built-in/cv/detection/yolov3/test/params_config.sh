#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

yolov3_base_params () {
    device="mlu"

    batch_size="16"
    seed="66"
    epochs="273"
    precision="fp32"
    num_workers="4"
    ddp="False"

    benchmark_mode="True"
    evaluate="False"
    max_batch_size_MLU290="16"
    max_batch_size_MLU370="16"
    max_batch_size_MLU590="16"
    max_batch_size_MLU370_ECC="16"
    max_batch_size_V100="16"

    yolo_path=$CONFIG_DIR/..
    data=$yolo_path/models/data/coco2014.data
    cfg=$yolo_path/models/cfg/yolov3.cfg
    ckp_dir=$yolo_path/models/weights/
    log_dir=$yolo_path/models/logs/
    weights=$PYTORCH_TRAIN_CHECKPOINT/yolov3/model_last.pth.tar
    resume="True"
}

set_configs () {
    params=$1

    # 调用相应网络的base_params
    yolov3_base_params

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
            ddp)    ddp="True" ;
                    batch_size=64 ;;
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

        if [[ $ddp == "True" ]]; then
           # yolov3脚本跑ddp时需要调整卡数
           get_visible_cards DEVICE_COUNT
           batch_size=$(($batch_size*$DEVICE_COUNT))
        fi

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi

    # 配置推理时的参数
    if [[ ${evaluate} == "True" ]]; then
        img_size=416;
        iou=0.6;
        task=test;
        weights=$yolo_path/models/weights/epoch_272.pth
        batch_size=32
    fi
}
