#!/bin/bash
# 该脚本功能为实现单MLU 验证集推理 输出精度
# 官方未提供checkpoint 使用mlu_trained_checkpoint SD_ResNet50_infer.py 采用复用 SD_ResNet50_train.py的方式
set -e
device='MLU'
epoch=65
SAVE_CHECKPOINT_PATH=$PROJ_DIR/data/mlu_trained_checkpoint/SSD_ResNet50_${epoch}E_FP32_4MLUs
SAVE_RESULT_PATH=$PROJ_DIR/data/output/eval/SSD_ResNet50_Eval
mkdir -p $SAVE_CHECKPOINT_PATH
mkdir -p $SAVE_RESULT_PATH

# Set Metric Envs
export MLU_VISIBLE_DEVICES=0,1,2,3
pushd $PROJ_DIR/models
python SSD_ResNet50_infer.py    --backbone resnet50 \
                                --backbone-path ${PYTORCH_TRAIN_CHECKPOINT}ssd/resnet50-19c8e357.pth \
                                --bs 32 \
                                --warmup 300 \
                                --mode evaluation \
                                --checkpoint $SAVE_CHECKPOINT_PATH/last.pt \
                                --data $COCO2017_TRAIN_DATASET \
                                --input_data_type float32 \
                                --json-summary $SAVE_RESULT_PATH/eval.json