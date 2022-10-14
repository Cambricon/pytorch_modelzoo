set -e
# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 IWSLT_CORPUS_PATH TRANSFORMER_CKPT [IMAGE_NAME] [CONTAINER_NAME]"
    echo "|      Supported options:"
    echo "|             IWSLT_CORPUS_PATH: dataset path."
    echo "|             TRANSFORMER_CKPT: ckpt path."
    echo "|             IMAGE_NAME: the docker image to run."
    echo "|             CONTAINER_NAME: container name."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

while getopts 'h:' opt; do
   case "$opt" in
       h)  usage ; exit 1 ;;
       ?)  echo "unrecognized optional arg : "; $opt; usage; exit 1;;
   esac
done
IMAGE_NAME=yellow.hub.cambricon.com/pytorch/pytorch:v1.6.0-torch1.6-ubuntu18.04
CONTAINER_NAME=mlu_transformer
if [ -z $IWSLT_CORPUS_PATH ]; then
  echo "please set environment variable IWSLT_CORPUS_PATH."
  exit 1
fi
if [ -z $TRANSFORMER_CKPT ]; then
  echo "please set environment variable TRANSFORMER_CKPT."
  exit 1
fi
if [[ $3 != "" ]]; then
    IMAGE_NAME=$3
fi
if [[ $4 != "" ]]; then
    CONTAINER_NAME=$4
fi
num=`docker ps -a|grep "$CONTAINER_NAME"|wc -l`
echo $num
echo $CONTAINER_NAME
if [ 0 -eq $num ]; then
docker run --device /dev/cambricon_ctl --network host --ipc=host -it --privileged --name $CONTAINER_NAME -v /data:/data -v /usr/bin/cnmon:/usr/bin/cnmon -e IWSLT_CORPUS_PATH=$IWSLT_CORPUS_PATH -e TRANSFORMER_CKPT=$TRANSFORMER_CKPT -v $PWD/../../..:/home/pytorch_modelzoo -w /home/pytorch_modelzoo/built-in/nlp/transformer/ $IMAGE_NAME /bin/bash
else
docker start $CONTAINER_NAME
docker exec -ti $CONTAINER_NAME /bin/bash
fi