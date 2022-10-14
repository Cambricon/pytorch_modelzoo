#!/bin/bash
#set -e

OS_NAME=NULL
declare -A device_dict

function ParseMLUID() {
    card_num=$(cnmon info | grep "Card" | wc -l)
    for ((card_id=0; card_id<$card_num; card_id++))
    do
        domain_id=$(cnmon info -c $card_id | grep "Domain ID" | awk -F ': ' '{print $2}' | sed 's/ //g')
        bus_num=$(cnmon info -c $card_id | grep "Bus num" | awk -F ': ' '{print $2}' | sed 's/ //g')
        device=$(cnmon info -c $card_id | grep "Device     " | awk -F ': ' '{print $2}' | sed 's/ //g')
        function=$(cnmon info -c $card_id | grep "Function" | awk -F ': ' '{print $2}' | sed 's/ //g')
        key=$domain_id:$bus_num:$device.$function
        device_dict["$key"]=$card_id
    done
}

function checkOs() {
  if [[ -f "/etc/lsb-release" ]];then
    OS_NAME=$(cat /etc/lsb-release | awk -F '=' '{if($1=="DISTRIB_ID") print $2}')
  elif [[ -f "/etc/redhat-release" ]];then
    OS_NAME="CentOS Linux"
  else
    echo -e "\033[31m ERROR: Only Support Ubuntu and CentOs.\033[0m"
    exit 1
  fi
}

function setCPUPerfMode() {
  if [ "$OS_NAME" == "Ubuntu" ]
  then
    installed_version=$(dpkg -l linux-tools-$(uname -r) | grep linux-tools-$(uname -r) | awk '{print $3}')
    sys_version=$(uname -r | awk -F '-generic' '{print $1}') bool_match=$(echo $installed_version | grep $sys_version)
    if [ "$bool_match" == "" ]
    then
      apt-get install -y linux-tools-$(uname -r)
    fi
  elif [ "$OS_NAME" == "CentOS Linux" ]
  then
    installed_version=$(cpupower -v | awk '{if(NR==1) print $2}' | awk -F '.debug' '{print $1}')
    sys_version=$(uname -r | awk -F '-generic' '{print $1}')
    bool_match=$(echo $installed_version | grep $sys_version)
    if [ "$bool_match" == "" ]
    then
      yum install cpupowerutils
    fi
  else
    echo -e "\033[31m ERROR: Set Performance Mode Failed. Only Support Ubuntu and CentOs. \033[0m"
    exit 1
  fi
  performance_mode=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
  if [ "$performance_mode" != "performance" ]
  then
    perf_cpu=$(cpupower -c all frequency-set -g performance)
    echo -e "\033[32m$perf_cpu \033[0m"
    # check performance mode
    performance_mode=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    if [ "$performance_mode" == "performance" ]
    then
      echo -e "\033[32m The CPU Performance Mode Enabled!\033[0m"
    else
      echo -e "\033[31m The CPU $performance_mode Mode Enabled! Please Check It.\033[0m"
      exit 1
    fi
  else
    echo -e "\033[32m The CPU $performance_mode Mode Enabled!\033[0m"
  fi
}

function set_task_accelerate() {
  if [ ! -d "/proc/driver/cambricon/mlus" ]
    then
      echo -e "\033[31m ERROR: No Exit /proc/driver/cambricon/mlus\033[0m"
      exit 1
  fi
  for mlu_file in `ls /proc/driver/cambricon/mlus`
  do
    bool_set_task_acc=$(cat /proc/driver/cambricon/mlus/$mlu_file/schedule_policy | awk -F ' ' '{print $4}')
    if [ "$bool_set_task_acc" == "acc" ]
    then
      echo -e "\033[32m Card ${device_dict[$mlu_file]} is task_accelerate.\033[0m"
    else
      $(echo acc > /proc/driver/cambricon/mlus/$mlu_file/schedule_policy)
      bool_set_task_acc=$(cat /proc/driver/cambricon/mlus/$mlu_file/schedule_policy | awk -F ' ' '{print $4}')
      if [ "$bool_set_task_acc" == "acc" ];then
        echo -e "\033[32m set Card ${device_dict[$mlu_file]} success.\033[0m"
      else
        echo -e "\033[31m ERROR: set Card ${device_dict[$mlu_file]} fail. the card ${device_dict[$mlu_file]} is busy.\033[0m"
      fi
    fi
  done
}

ParseMLUID
checkOs
setCPUPerfMode
set_task_accelerate
