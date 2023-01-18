set -e
# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 "
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

while getopts 'h:' opt; do
   case "$opt" in
       h)  usage ; exit 1 ;;
       ?)  echo "unrecognized optional arg : "; $opt; usage; exit 1;;
   esac
done
IMAGE_NAME=yellow.hub.cambricon.com/pytorch/pytorch:v1.8.0-torch1.6-ubuntu18.04-py37
CONTAINER_NAME=mlu_fasterrcnn

num=`docker ps -a|grep "$CONTAINER_NAME"|wc -l`
echo $num
echo $CONTAINER_NAME
if [ 0 -eq $num ]; then
# datasets coco2017
docker run  --device /dev/cambricon_ctl --network host --pid=host \
            -v /sys/kernel/debug:/sys/kernel/debug -it --privileged --ipc=host \
            --name $CONTAINER_NAME \
            -v $COCO2017_TRAIN_DATASET:$COCO2017_TRAIN_DATASET \
            -v $PYTORCH_TRAIN_CHECKPOINT:$PYTORCH_TRAIN_CHECKPOINT \
            -e COCO2017_TRAIN_DATASET=$COCO2017_TRAIN_DATASET \
            -e PYTORCH_TRAIN_CHECKPOINT=$PYTORCH_TRAIN_CHECKPOINT \
            -v $PWD/../../../..:/home/pytorch_modelzoo  \
            -w /home/pytorch_modelzoo/built-in/cv/detection/FasterRCNN/ \
            $IMAGE_NAME /bin/bash
else
docker start $CONTAINER_NAME
docker exec -ti $CONTAINER_NAME /bin/bash
fi
