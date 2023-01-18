#!/bin/bash
set -e
export MLU_VISIBLE_DEVICES=0
pushd $PROJ_DIR/models
python MaskRCNN_infer.py    --config-file configs/MaskRCNN_Eval.yaml \
                            --ckpt $PROJ_DIR/data/weights/gpu_checkpoints/e2e_mask_rcnn_R_101_FPN_1x.pth \
                            MODEL.DEVICE mlu \
                            TEST.IMS_PER_BATCH 16