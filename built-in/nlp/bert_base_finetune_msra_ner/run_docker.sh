#/bin/bash

export MY_CONTAINER="bert_base_finetune_msra_ner_pytorch_1_6_0"
IMAGE_NAME=yellow.hub.cambricon.com/pytorch/pytorch:v1.8.0-torch1.6-ubuntu18.04-py37 # use latest cambricon-pytorch image
num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
echo $MY_CONTAINER
if [ 0 -eq $num ];then
xhost +
docker run \
    --shm-size '64gb' \
    --privileged --network=host --cap-add=sys_ptrace --pid=host -P \
    --device /dev/cambricon_ctl \
    -it --name $MY_CONTAINER \
    -v path_of_pytorch_modelzoo:path_of_pytorch_modelzoo \
    -v /usr/bin/cnmon:/usr/bin/cnmon \
    $IMAGE_NAME \
    /bin/bash
else
docker start $MY_CONTAINER
docker exec -ti $MY_CONTAINER /bin/bash
fi
