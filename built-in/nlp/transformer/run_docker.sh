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
CONTAINER_NAME=mlu_transformer
if [ -z $IMAGE_NAME ]; then
  echo "please set environment variable IMAGE_NAME."
  exit 1
fi
if [ -z $IWSLT_CORPUS_PATH ]; then
  echo "please set environment variable IWSLT_CORPUS_PATH."
  exit 1
fi
if [ -z $TRANSFORMER_CKPT ]; then
  echo "please set environment variable TRANSFORMER_CKPT."
  exit 1
fi
if [[ $3 != "" ]]; then
    CONTAINER_NAME=$3
fi
num=`docker ps -a|grep "$CONTAINER_NAME"|wc -l`
echo $num
echo $CONTAINER_NAME
if [ 0 -eq $num ]; then
# change the `-v /your/data:/your/data` and `-v /your/ckpt:/your/ckpt`  to your data and checkpoint volume
docker run --device /dev/cambricon_ctl --network host --ipc=host -it --privileged --name $CONTAINER_NAME -v /your/data:/your/data -v /your/ckpt:/your/ckpt -v /usr/bin/cnmon:/usr/bin/cnmon -e IWSLT_CORPUS_PATH=$IWSLT_CORPUS_PATH -e TRANSFORMER_CKPT=$TRANSFORMER_CKPT -v $PWD/../../..:/home/pytorch_modelzoo -w /home/pytorch_modelzoo/built-in/nlp/transformer/ $IMAGE_NAME /bin/bash
else
docker start $CONTAINER_NAME
docker exec -ti $CONTAINER_NAME /bin/bash
fi
