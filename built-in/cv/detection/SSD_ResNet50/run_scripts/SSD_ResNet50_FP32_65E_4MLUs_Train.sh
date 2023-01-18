#!/bin/bash
# 该脚本功能为实现4MLUs FP32训练
set -e
epoch=65 #default
SAVE_CHECKPOINT_PATH=$PROJ_DIR/data/mlu_trained_checkpoint/SSD_ResNet50_${epoch}E_FP32_4MLUs
SAVE_RESULT_PATH=$PROJ_DIR/data/output/train/SSD_ResNet50_${epoch}E_FP32_4MLUs
mkdir -p $SAVE_CHECKPOINT_PATH
mkdir -p $SAVE_RESULT_PATH

export MLU_VISIBLE_DEVICES=0,1,2,3
pushd $PROJ_DIR/models
python -m torch.distributed.launch  --nproc_per_node=4 SSD_ResNet50_train.py  \
                                    --backbone resnet50 \
                                    --backbone-path ${PYTORCH_TRAIN_CHECKPOINT}ssd/resnet50-19c8e357.pth \
                                    --bs 32 \
                                    --warmup 300 \
                                    --save $SAVE_CHECKPOINT_PATH \
                                    --data $COCO2017_TRAIN_DATASET \
                                    --iterations -1 \
                                    --epochs $epoch \
                                    --json-summary $SAVE_RESULT_PATH/train.json