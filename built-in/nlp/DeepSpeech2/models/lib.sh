#!/bin/bash
# set -e

OS_NAME=NULL

if [[ -f "/etc/lsb-release" ]];then
  OS_NAME=$(cat /etc/lsb-release | awk -F '=' '{if($1=="DISTRIB_ID") print $2}')
elif [[ -f "/etc/redhat-release" ]];then
  OS_NAME="CentOS Linux"
else
  echo -e "\033[31m ERROR: Only Support Ubuntu and CentOs.\033[0m"
fi

if [ "$OS_NAME" == "Ubuntu" ]
then
  apt-get -y install apt-utils libpq-dev libsndfile-dev
  apt-get -y install sox
elif [ "$OS_NAME" == "CentOS Linux" ]
then
  yum -y install libsndfile
  yum -y install sox
else
  echo -e "\033[31m ERROR: Set Performance Mode Failed. Only Support Ubuntu and CentOs. \033[0m"
  exit 1
fi
