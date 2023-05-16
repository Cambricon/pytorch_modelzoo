export MY_CONTAINER="mlu_mt5_pytorch_1_9"
IMAGE_NAME=yellow.hub.cambricon.com/pytorch/pytorch:v1.11.0-torch1.9-ubuntu18.04-py37 # use latest cambricon-pytorch image
num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
echo $MY_CONTAINER
if [ 0 -eq $num ];then
xhost +
docker run \
    --shm-size '64gb' \
    --privileged --network=host --ipc=host \
    --device /dev/cambricon_ctl \
    -it --name $MY_CONTAINER \
    -v $PWD/../../..:/home/pytorch_modelzoo \
    -v /your/data:/your/data \
    -v /usr/bin/cnmon:/usr/bin/cnmon \
    -w /home/pytorch_modelzoo/built-in/nlp/mt5/ \
    $IMAGE_NAME \
    /bin/bash
else
docker start $MY_CONTAINER
docker exec -ti $MY_CONTAINER /bin/bash
fi
