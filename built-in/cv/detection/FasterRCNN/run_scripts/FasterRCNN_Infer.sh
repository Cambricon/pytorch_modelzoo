#!/bin/bash
set -e
export MLU_VISIBLE_DEVICES=0
pushd $PROJ_DIR/models
# model_final.pth基于FasterRCNN_FP32_20000S_8MLUs_Train.sh训练得到，路径见脚本内部指定
python FasterRCNN_infer.py  --config-file configs/FasterRCNN_Eval.yaml \
                            --ckpt $PROJ_DIR/data/weights/gpu_checkpoints/e2e_faster_rcnn_R_101_FPN_1x.pth \
                            MODEL.DEVICE mlu \
                            TEST.IMS_PER_BATCH 16