#!/bin/bash
set -e

EPOCH=300
SAVE_PATH=$PROJ_DIR/data/output/train/MLU_RFBNet_${EPOCH}S_FP32_4MLUs
if  [ ! -d $SAVE_PATH ];then
    mkdir -p $SAVE_PATH
fi

export MASTER_ADDR='127.0.0.1'
export MASTER_PORT=28883
export OMP_NUM_THREADS=1  
export MLU_VISIBLE_DEVICES=0,1,2,3

pushd $PROJ_DIR/models
# train
python RFBNet_train.py -d VOC -v RFB_vgg -s 300\
        --device mlu \
        --distributed \
        --nprocessor 4 \
        --batch_size 32 \
        --save_folder $SAVE_PATH \
         -max $EPOCH \
        --world_size 1 \
        --node_rank 0  \
        --mode scratch \
        --basenet ${PYTORCH_TRAIN_CHECKPOINT}rfbnet/checkpoints_fp/vgg16_reducedfc.pth

# eval after train
python RFBNet_infer.py -d VOC -v RFB_vgg -s 300 --device mlu --trained_model $SAVE_PATH/Final_RFB_vgg_VOC.pth