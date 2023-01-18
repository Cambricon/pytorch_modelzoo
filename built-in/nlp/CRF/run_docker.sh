set -e
# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 [CONTAINER_NAME]"
    echo "|      Supported options:"
    echo "|             CONTAINER_NAME: container name."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

while getopts 'h:' opt; do
   case "$opt" in
       h)  usage ; exit 1 ;;
       ?)  echo "unrecognized optional arg : "; $opt; usage; exit 1;;
   esac
done
CONTAINER_NAME=mlu_CRF
if [ -z $IMAGE_NAME ]; then
  echo "please set environment variable IMAGE_NAME."
  exit 1
fi
if [ -z $CRF_DATASET ]; then
  echo "please set environment variable CRF_DATASET."
  exit 1
fi

if [[ $1 != "" ]]; then
      CONTAINER_NAME=$1
fi

num=`docker ps -a|grep "$CONTAINER_NAME"|wc -l`
echo $num
echo $CONTAINER_NAME

if [ 0 -eq $num ]; then
docker run \
    --device /dev/cambricon_ctl \
    --network host \
    --ipc=host \
    -v /usr/bin/cnmon:/usr/bin/cnmon -it \
    --privileged \
    --name $CONTAINER_NAME \
    -e CRF_DATASET=$CRF_DATASET \
    -v $PWD/../../../:/home/pytorch_modelzoo/ \
    -w /home/pytorch_modelzoo/built-in/nlp/CRF/ $IMAGE_NAME /bin/bash
else
docker start $CONTAINER_NAME
docker exec -ti $CONTAINER_NAME /bin/bash
fi