#!/bin/bash
set -e

ITERATIONS=20000
SAVE_PATH=$PROJ_DIR/data/output/train/MLU_FasterRCNN_${ITERATIONS}S_FP32
if  [ ! -d $SAVE_PATH ];then
    mkdir -p $SAVE_PATH
fi

export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DEVICES_NUMS=$(echo $MLU_VISIBLE_DEVICES | awk -F "," '{print NF}')
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
pushd $PROJ_DIR/models
python -m torch.distributed.launch  --master_addr=$MASTER_ADDR \
                                    --nproc_per_node=$DEVICES_NUMS \
                                    --master_port=$PORT \
                                    FasterRCNN_train.py \
                                    --config-file configs/FasterRCNN_FP32_Train.yaml \
                                    --prefix faster \
                                    MODEL.DEVICE mlu \
                                    MODEL.WEIGHT ${PYTORCH_TRAIN_CHECKPOINT}rcnn/basenet/R-101.pkl