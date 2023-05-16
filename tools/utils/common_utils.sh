#!/bin/bash


function get_platform () {
    set +e
    platform=$1
    mlu_model=`cat /proc/driver/*/*/*/information | grep "Device name" | uniq`
    gpu_model=`cat /proc/driver/*/*/*/information | grep "Model" | uniq`
    if [[ $mlu_model ]]; then
        platform=`echo $mlu_model | awk -F " " '{print $3}' | awk -F "-" '{print $1}'`
    elif [[ $gpu_model ]]; then
        platform=`echo $gpu_model | awk -F ":" '{print $2}' | awk -F "-" '{print $1}' | awk -F " " '{print $2}'`
    fi
    echo "Platform is: " $platform
    eval $1=$platform
    set -e
}

function get_platform_with_flag_name () {
    set +e
    platform=$1
    mlu_model=`cat /proc/driver/*/*/*/information | grep "Device name" | uniq`
    gpu_model=`cat /proc/driver/*/*/*/information | grep "Model" | uniq`
    if [[ $mlu_model ]]; then
        platform=`echo $mlu_model | awk -F " " '{print $3}'`
        platform=${platform/-/_}
        str_contain=$(echo ${platform} | grep "MLU370")
        if [[ ${str_contain} != "" ]]; then
            platform="MLU370"
        fi
    elif [[ $gpu_model ]]; then
        platform=`echo $gpu_model | awk -F ":" '{print $2}' | awk -F "-" '{print $1}' | awk -F " " '{print $2}'`
    fi
    echo "Platform is: " $platform
    eval $1=$platform
    set -e
}

function get_ecc_status () {
    set +e
    ecc_status=$1
    ecc_status="OFF"
    mlu_model=`cat /proc/driver/*/*/*/information | grep "Device name" | uniq`
    platform=`echo $mlu_model | awk -F " " '{print $3}' | awk -F "-" '{print $1}'`
    if [[ $mlu_model ]]; then
        # Note: Only MLU-370 needs to speicify ECC ON and ECC off, other platform only support ECC ON
        ecc_status=`cnmon info | grep "DDR ECC Err Count" | uniq | awk -F ":" '{print $2}'`
        if [[ $platform == "MLU370" && $ecc_status != "N/A" ]]; then
            ecc_status="ON"
        fi
        echo "ECC status: " $ecc_status
    fi
    eval $1=$ecc_status
    set -e
}

function perf_iters_rule () {
    set +e
    cur_platform="unkonw"
    get_platform cur_platform

    total_iters=-1
    cutdown_iters=-1
    if [[ $cur_platform == "MLU290" ]]; then
        echo "cur_platform is mlu290 ... "
        total_iters=300
        cutdown_iters=100
    else
        total_iters=60
        cutdown_iters=20
    fi
    if [[ $CATCH_MODEL_TOTAL_ITERS ]]; then
        total_iters=`echo $CATCH_MODEL_TOTAL_ITERS`
        cutdown_iters=1
        echo "SET CATCH_MODEL_TOTAL_ITERS for internal test, currently run ${total_iters} iters."
    fi
    eval $1=$total_iters
    export MLU_ADAPTIVE_STRATEGY_COUNT=$cutdown_iters
    set -e
}

function get_visible_cards () {
    set +e
    mlu_model=`cat /proc/driver/*/*/*/information | grep "Device name" | uniq`
    gpu_model=`cat /proc/driver/*/*/*/information | grep "Model" | uniq`
    if [[ $mlu_model ]]; then
        if [[ $MLU_VISIBLE_DEVICES ]]; then
            cards_num=`echo $MLU_VISIBLE_DEVICES | awk -F, '{print NF}'`
        else
            echo "MLU_VISIBLE_DEVICES unset, please set it before running multicards."
            cards_num=-1
        fi
    elif [[ $gpu_model ]]; then
        if [[ $CUDA_VISIBLE_DEVICES ]]; then
            cards_num=`echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}'`
        else
            echo "CUDA_VISIBLE_DEVICES unset, please set it before running multicards."
            cards_num=-1
        fi
    else
        echo "function get_visible_cards can only used on MLU or GPU platform."
        cards_num=0
    fi
    eval $1=$cards_num
    set -e
}

