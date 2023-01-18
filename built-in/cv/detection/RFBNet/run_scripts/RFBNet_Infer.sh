#!/bin/bash
set -e

EPOCH=300
SAVE_PATH=$PROJ_DIR/data/output/train/MLU_RFBNet_${EPOCH}S_AMP_4MLUs
if  [ ! -d $SAVE_PATH ];then
    mkdir -p $SAVE_PATH
fi

export MASTER_ADDR='127.0.0.1'
export MASTER_PORT=28882
export OMP_NUM_THREADS=1  
export MLU_VISIBLE_DEVICES=0
pushd $PROJ_DIR/models
# 推理使用基于Train.sh训练得到的模型，模型存放于$SAVE_PATH下。
python RFBNet_infer.py -d VOC -v RFB_vgg -s 300 --device mlu --trained_model $SAVE_PATH/Final_RFB_vgg_VOC.pth

