#!/bin/bash
set -e

export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DEVICES_NUMS=$(echo $MLU_VISIBLE_DEVICES | awk -F "," '{print NF}')
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
pushd $PROJ_DIR/models
python -m torch.distributed.launch  --master_addr=$MASTER_ADDR \
                                    --nproc_per_node=$DEVICES_NUMS \
                                    --master_port=$PORT \
                                    MaskRCNN_train.py \
                                    --config-file configs/MaskRCNN_FP32_8MLUs_Train.yaml \
                                    --prefix mask \
                                    MODEL.DEVICE mlu \
                                    MODEL.WEIGHT ${PYTORCH_TRAIN_CHECKPOINT}rcnn/basenet/R-101.pkl \
                                    MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000
