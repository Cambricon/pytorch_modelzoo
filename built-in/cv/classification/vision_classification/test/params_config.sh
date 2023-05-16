#!/bin/bash

#IMAGENET_TRAIN_CHECKPOINT=/data/pytorch/models/pytorch_weight/checkpoints/

CONFIG_DIR=$(cd $(dirname $0);pwd)

resnet50_base_params () {
    net="resnet50"
    device="mlu"

    batch_size="64"
    lr="0.025"
    weight_decay="1e-4"
    momentum="0.9"
    seed="42"
    epochs=75
    precision="fp32"
    num_workers="12"
    ddp="False"

    benchmark_mode="True"
    max_batch_size_MLU290="256"
    max_batch_size_MLU370="256"
    max_batch_size_MLU590_M9="792"
    max_batch_size_MLU590_M9U="792"
    max_batch_size_MLU590_H8="696"
    max_batch_size_MLU370_ECC="224"
    max_batch_size_V100="128"

    resume="${IMAGENET_TRAIN_CHECKPOINT}/resnet50/epoch_74.pth"
}

resnet18_base_params () {
    net="resnet18"
    device="mlu"

    batch_size="128"
    lr="0.05"
    weight_decay="1e-4"
    momentum="0.9"
    seed="42"
    epochs=75
    precision="fp32"
    num_workers="32"
    ddp="False"

    benchmark_mode="True"
    max_batch_size_MLU290="512"
    max_batch_size_MLU370="664"
    max_batch_size_MLU590_M9="2048"
    max_batch_size_MLU590_M9U="2048"
    max_batch_size_MLU590_H8="2560"
    max_batch_size_MLU370_ECC="664"
    max_batch_size_V100="512"

    resume="${IMAGENET_TRAIN_CHECKPOINT}/resnet18/epoch_74.pth"
}

vgg16_base_params(){
    net="vgg16"
    device="mlu"

    batch_size="128"
    lr="0.005"
    weight_decay="1e-4"
    momentum="0.9"
    seed="42"
    epochs=50
    precision="fp32"
    num_workers="8"
    ddp="False"

    benchmark_mode="True"
    max_batch_size_MLU290="128"
    max_batch_size_MLU370="128"
    max_batch_size_MLU590_M9="664"
    max_batch_size_MLU590_M9U="664"
    max_batch_size_MLU590_H8="664"
    max_batch_size_MLU370_ECC="128"
    max_batch_size_V100="128"

    resume="${IMAGENET_TRAIN_CHECKPOINT}/vgg16/epoch_49.pth"
}

vgg19_base_params(){
    net="vgg19"
    device="mlu"

    batch_size="64"
    lr="0.0025"
    weight_decay="1e-4"
    momentum="0.9"
    seed="42"
    epochs=50
    precision="fp32"
    num_workers="8"
    ddp="False"

    benchmark_mode="True"
    max_batch_size_MLU290="128"
    max_batch_size_MLU370="64"
    max_batch_size_MLU590_M9="664"
    max_batch_size_MLU590_M9U="664"
    max_batch_size_MLU590_H8="664"
    max_batch_size_MLU370_ECC="64"
    max_batch_size_V100="64"

    resume="${IMAGENET_TRAIN_CHECKPOINT}/vgg19/epoch_49.pth"
}

vgg16_bn_base_params(){
    net="vgg16_bn"
    device="mlu"

    batch_size="32"
    lr="0.01"
    weight_decay="1e-4"
    momentum="0.9"
    seed="42"
    epochs=50
    precision="fp32"
    num_workers="8"
    ddp="False"

    benchmark_mode="True"
    max_batch_size_MLU290="128"
    max_batch_size_MLU370="128"
    max_batch_size_MLU590_M9="664"
    max_batch_size_MLU590_M9U="664"
    max_batch_size_MLU590_H8="504"
    max_batch_size_MLU370_ECC="128"
    max_batch_size_V100="128"

    resume="${IMAGENET_TRAIN_CHECKPOINT}/vgg16_bn/epoch_49.pth"
}

mobilenet_v2_base_params () {
    net="mobilenet_v2"
    device="mlu"

    batch_size="128"
    lr="0.025"
    weight_decay="1e-4"
    momentum="0.9"
    seed="42"
    epochs=50
    precision="fp32"
    num_workers="8"
    ddp="False"

    benchmark_mode="True"
    max_batch_size_MLU290="256"
    max_batch_size_MLU370="256"
    max_batch_size_MLU590_M9="1280"
    max_batch_size_MLU590_M9U="1280"
    max_batch_size_MLU590_H8="1024"
    max_batch_size_MLU370_ECC="256"
    max_batch_size_V100="128"

    resume="${IMAGENET_TRAIN_CHECKPOINT}/mobilenetv2/epoch_49.pth"
}

alexnet_base_params () {
    net="alexnet"
    device="mlu"

    batch_size="256"
    lr="0.01"
    weight_decay="1e-4"
    momentum="0.9"
    seed="42"
    epochs=50
    precision="fp32"
    num_workers="12"
    ddp="False"

    benchmark_mode="True"
    max_batch_size_MLU290="256"
    max_batch_size_MLU370="512"
    max_batch_size_MLU590_M9="1024"
    max_batch_size_MLU590_M9U="1024"
    max_batch_size_MLU590_H8="1024"
    max_batch_size_MLU370_ECC="512"
    max_batch_size_V100="256"

    resume="${IMAGENET_TRAIN_CHECKPOINT}/alexnet/epoch_49.pth"
}

resnet101_base_params () {
    net="resnet101"
    device="mlu"

    batch_size="64"
    epochs=50
    lr="0.025"
    weight_decay="4e-05"
    momentum="0.9"
    seed="42"
    precision="fp32"
    num_workers="8"
    ddp="False"

    benchmark_mode="True"
    max_batch_size_MLU290="128"
    max_batch_size_MLU370="168"
    max_batch_size_MLU590_M9="664"
    max_batch_size_MLU590_M9U="664"
    max_batch_size_MLU590_H8="504"
    max_batch_size_MLU370_ECC="136"
    max_batch_size_V100="128"

    resume="${IMAGENET_TRAIN_CHECKPOINT}/resnet101/epoch_49.pth"
}

shufflenet_v2_x0_5_base_params () {
    net="shufflenet_v2_x0_5"
    device="mlu"

    batch_size="1024"
    epochs=151
    lr="0.025"
    weight_decay="4e-5"
    momentum="0.9"
    seed="42"

    precision="fp32"
    num_workers="8"
    ddp="False"

    benchmark_mode="True"
    max_batch_size_MLU290="128"
    max_batch_size_MLU370="1024"
    max_batch_size_MLU590_M9="1280"
    max_batch_size_MLU590_M9U="1280"
    max_batch_size_MLU590_H8="2560"
    max_batch_size_MLU370_ECC="784"
    max_batch_size_V100="1024"

    resume="${IMAGENET_TRAIN_CHECKPOINT}/shufflenet_v2_x0_5/epoch_150.pth"
}

shufflenet_v2_x1_0_base_params () {
    net="shufflenet_v2_x1_0"
    device="mlu"

    batch_size="512"
    epochs=151
    lr="0.0625"
    weight_decay="4e-5"
    momentum="0.9"
    seed="42"

    precision="fp32"
    num_workers="8"
    ddp="False"

    benchmark_mode="True"
    max_batch_size_MLU290="128"
    max_batch_size_MLU370="976"
    max_batch_size_MLU590_M9="1280"
    max_batch_size_MLU590_M9U="1280"
    max_batch_size_MLU590_H8="2560"
    max_batch_size_MLU370_ECC="784"
    max_batch_size_V100="512"

    resume="${IMAGENET_TRAIN_CHECKPOINT}/shufflenet_v2_x1_0/epoch_150.pth"
}

shufflenet_v2_x1_5_base_params () {
    net="shufflenet_v2_x1_5"
    device="mlu"

    batch_size="256"
    epochs=151
    lr="0.03125"
    weight_decay="4e-5"
    momentum="0.9"
    seed="42"

    precision="fp32"
    num_workers="8"
    ddp="False"

    benchmark_mode="True"
    max_batch_size_MLU290="128"
    max_batch_size_MLU370="512"
    max_batch_size_MLU590_M9="1024"
    max_batch_size_MLU590_M9U="1024"
    max_batch_size_MLU590_H8="2048"
    max_batch_size_MLU370_ECC="512"
    max_batch_size_V100="256"

    resume="${IMAGENET_TRAIN_CHECKPOINT}/shufflenet_v2_x1_5/epoch_150.pth"
}

googlenet_base_params () {
    net="googlenet"
    device="mlu"

    batch_size="64"
    lr="0.025"
    weight_decay="4e-05"
    momentum="0.9"
    seed="42"
    epochs=31
    precision="fp32"
    num_workers="8"
    ddp="False"

    benchmark_mode="True"
    max_batch_size_MLU290="128"
    max_batch_size_MLU370="256"
    max_batch_size_MLU590_M9="1560"
    max_batch_size_MLU590_M9U="1560"
    max_batch_size_MLU590_H8="1024"
    max_batch_size_MLU370_ECC="256"
    max_batch_size_V100="128"

    resume="${IMAGENET_TRAIN_CHECKPOINT}/googlenet/epoch_30.pth"
}

set_configs () {
    args=$1

    # 获取网络和参数字段
    net=${args%%-*}
    params=${args#*-}

    # 调用相应网络的base_params
    ${net}_base_params

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
                       resume_multi_device="True";
                       ;;
            ci_eval)   benchmark_mode=False;
                       iters=2;
                       resume_multi_device="True";
                       evaluate="True";
                       ;;
	    dummy_test) dummy_test="True" ;;
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
        get_platform_with_flag_name cur_platform
        mbs_name=max_batch_size_${cur_platform}

        cur_ecc_status=""
        get_ecc_status cur_ecc_status
        if [[ ${cur_ecc_status} == "ON" ]]; then
            mbs_name=max_batch_size_${cur_platform}_ECC
        fi
        batch_size=${!mbs_name}

        epochs=1
        resume=""

        visible_cards=-1
        get_visible_cards visible_cards
        ## 检查多卡时是否设置VISIBLE_DEVICES环境变量
        if [[ $ddp == "True" ]]; then
            if [ $visible_cards -eq -1 ]; then
                # echo "Please set env MLU_VISIBLE_DEVICES before running multicards."
                exit 1
            fi
            if [[ ${net} == "shufflenet_v2_x0_5" ]]; then
                echo "Set OPENBLAS_NUM_THREADS=2"
                export OPENBLAS_NUM_THREADS=2
            fi
        fi

        ## set num_workers for different platforms and cardnums
        num_workers="32"
        if [[ ${visible_cards} -eq 8 ]]; then
            num_workers="12"
        fi
        if [[ ${cur_platform} == "MLU370" ]]; then
            if [[ ${visible_cards} -eq 16 ]]; then
                num_workers="7"
            elif [[ ${visible_cards} -eq 4 ]]; then
                num_workers="16"
            fi
        fi

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi
}

