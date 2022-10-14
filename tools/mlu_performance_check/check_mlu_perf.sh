#!/bin/bash
# set -e

function print_log() {
  echo -e "\033[31m ERROR: Please execute the command: cd pytorch_models/Training/tools/mlu_performance_check; sudo bash set_mlu_perf.sh\033[0m"
}

function check_mount() {
  dataset_path=$1
  dataset_first_path=$(echo $dataset_path | awk -F '/' '{print $2}')
  all_mount_paths=$(df -i | awk '{if(NR>1)print $6}')
  for path in $all_mount_paths
  do
    first_path=$(echo $path | awk -F '/' '{print $2}')
    if [[ $dataset_path =~ $path && $dataset_first_path == $first_path ]]
    then
      echo -e "\033[34m WARNING: The $dataset_path is mounted, it may degrade performance.\033[0m"
      return 1
      break
    fi
  done
  return 0
}

function check_dataset_path() {
  check_mount $IMAGENET_PATH
  check_mount $VOC2012_PATH_PYTORCH
  check_mount $VOC2007_PATH_PYTORCH
  check_mount $COCO_PATH_PYTORCH
  check_mount $FDDB_PATH_PYTORCH
  check_mount $IMAGENET_PATH_PYTORCH
  check_mount $ICDAR_PATH_PYTORCH
  check_mount $VOC_DEVKIT
}


OS_NAME=NULL
function checkOs() {
  if [[ -f "/etc/lsb-release" ]];then
    OS_NAME=$(cat /etc/lsb-release | awk -F '=' '{if($1=="DISTRIB_ID") print $2}')
  elif [[ -f "/etc/redhat-release" ]];then
    OS_NAME="CentOS Linux"
  else
    echo -e "\033[31m ERROR: Only Support Ubuntu and CentOs.\033[0m"
  fi
}

function Is_X8_Or_MLU290() {
  card_id=$1
  bool_is_x8=$(cnmon -c $card_id | grep "X8")
  bool_is_mlu290=$(cnmon -c $card_id | grep "MLU290")
  if [[ "$bool_is_x8" == "" && "$bool_is_mlu290" == "" ]]
  then
    return 0
  else
    return 1
  fi
}

function checkCPUInfo() {
    cpu_model=$(cat /proc/cpuinfo | awk -F ':' '{if ($1 ~ "model name") print $2}' | uniq)
    cpu_physical_core_num=$(cat /proc/cpuinfo |grep "physical id"|sort|uniq | wc -l)
    processor_num=$(cat /proc/cpuinfo | grep "processor" | wc -l)
    echo -e "\033[32m$cpu_model\033[0m"
    echo -e "\033[32m CPU Physical Core Nums: $cpu_physical_core_num\033[0m"
    echo -e "\033[32m CPU Processor Nums: $processor_num\033[0m"
}

function FindCPUProcess() {
  info=$(top -n 1 | head -n 20 | awk '{if (NR>7 && NF==14 && $10>5) print $2, $10, $13; else if (NR>7 && NF==13 && $9>5) print $1, $9, $12}' | awk '{if ($3!="top") printf(" ERROR: The PID: %10s, COMMAND: %20s, Please Kill It.\n", $1, $3)}')
  if [ "$info" != "" ]
  then
    echo -e "\033[31m$info \033[0m"
  else
    echo -e "\033[32m No Programs Occupied CPUs!\033[0m"
  fi
}

function FindMLUPID() {
  line_num=$(cnmon | awk '{if($0 ~ "PID") print NR}')
  pid_info=$(cnmon | awk -v line=$line_num '{a=line;if(NR>a && NF == 8) printf(" ERROR: The PID %s Running On MLU, Please Kill It.\n", $4)}')
  if [ "$pid_info" != "" ]
  then
    echo -e "\033[31m$pid_info \033[0m"
  fi
}


function setEnv() {
  if [ -z "${NEUWARE_HOME}" ]
  then
    echo -e "\033[31m ERROR : NEUWARE_HOME is not set\033[0m"
  fi
  if [ -z "${CATCH_HOME}"  ]
  then
    echo -e "\033[31m ERROR : CATCH_HOME is not set\033[0m"
  fi
  if [ -z "${PYTORCH_HOME}"  ]
  then
    echo -e "\033[31m ERROR : PYTORCH_HOME is not set\033[0m"
  fi
  if [ -z "${VISION_HOME}"  ]
  then
    echo -e "\033[31m ERROR : VISION_HOME is not set\033[0m"
  fi
}

function checkNeuwareVersion() {
  pushd $CATCH_HOME/script/release/
    python json_parser.py
    neuware_lib_version=$(awk -F ':' '{if(($1 !~ "cntoolkit")&&($1 !~ "cnplugin")&&($1 !~ "cnlight")&&($1 !~ "magicmind")) print $1, $2, $3}' dependency.txt | awk '{printf "lib%s.so.%s\n", $1, $3}' | awk -F '-' '{printf "%s\n", $1}')
  popd
  for lib in $neuware_lib_version
  do
    if [ ! -f "$NEUWARE_HOME/lib64/$lib" ]
    then
      echo -e "\033[31m ERROR: No Exit $NEUWARE_HOME/lib64/$lib\033[0m"
    fi
  done
}

function checkMLULink() {
  max_card_num=16
  index_end=($max_card_num - 1)
  for i in `seq 0 $index_end`
  do
      bool_card_exit=$(cnmon info -c $i | grep "not existed")
      if [ "$bool_card_exit" != "" ]
      then
        continue
      fi
      # check X8 exit or not
      Is_X8_Or_MLU290 $i
      bool_exit=$?
      if [ $bool_exit == 0 ]
      then
        echo -e "\033[34m WARNING: This CardID $i isn't X8 or MLU290\033[0m"
        continue
      fi
      bool_mlulink=$(cnmon mlulink -c $i -s | awk -F ':' '{if(NF == 2) print $2}' | grep "Disable")
      if [ "$bool_mlulink" != "" ]
      then
        echo -e "\033[31m ERROR: MLUlink Failed. Please Check Card $i\033[0m"
      fi
  done
  echo -e "\033[32m MLUlink Success.\033[0m"
}

function check_task_accelerate() {
  bool_task_accelarate=1
  if [ ! -d "/proc/driver/cambricon/mlus" ]
    then
      echo -e "\033[31m ERROR: No Exit /proc/driver/cambricon/mlus\033[0m"
  fi
  for mlu_file in `ls /proc/driver/cambricon/mlus`
  do
    bool_set_task_acc=$(cat /proc/driver/cambricon/mlus/$mlu_file/schedule_policy | awk -F ' ' '{print $4}')
    if [ "$bool_set_task_acc" != "acc" ]
    then
      bool_task_accelarate=0
      echo -e "\033[31m ERROR: $mlu_file is not task_accelerate.\033[0m"
      print_log
    fi
  done
  if [ $bool_task_accelarate -eq 1 ];then
    echo -e "\033[32m MLU task_accelarete enabled successfully.\033[0m"
  fi
}


function checkCPUPerfMode() {
  if [ "$OS_NAME" == "Ubuntu" ]
  then
    installed_version=$(dpkg -l linux-tools-$(uname -r) | grep linux-tools-$(uname -r) | awk '{print $3}')
    sys_version=$(uname -r | awk -F '-generic' '{print $1}')
    bool_match=$(echo $installed_version | grep $sys_version)
    if [ "$bool_match" == "" ]
    then
      echo -e "\033[31m ERROR: Mismatch linux-tools-$(uname -r)\033[0m"
      print_log
    fi
  elif [ "$OS_NAME" == "CentOS Linux" ]
  then
    installed_version=$(cpupower -v | awk '{if(NR==1) print $2}' | awk -F '.debug' '{print $1}')
    sys_version=$(uname -r | awk -F '-generic' '{print $1}')
    bool_match=$(echo $installed_version | grep $sys_version)
    if [ "$bool_match" == "" ]
    then
      echo -e "\033[31m ERROR: Mismatch cpupower\033[0m"
      print_log
    fi
  else
    echo -e "\033[31m ERROR: Set Performance Mode Failed. Only Support Ubuntu and CentOs. \033[0m"
  fi
  performance_mode=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
  if [ "$performance_mode" != "performance" ]
  then
    echo -e "\033[31m ERROR: The CPU $performance_mode Mode Enabled!\033[0m"
    print_log
  else
    echo -e "\033[32m The CPU performance Mode Enabled!\033[0m"
  fi
}

check_dataset_path
setEnv
checkNeuwareVersion
checkOs
checkCPUInfo
FindCPUProcess
FindMLUPID
checkMLULink
check_task_accelerate
checkCPUPerfMode
