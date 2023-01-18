# GPU Eval Scripts
CUR_DIR=$(cd $(dirname $0);pwd)

export MLU_VISIBLE_DEVICES=0
SAVE_PATH=$PROJ_DIR/data/output/eval/GPU_SSD_VGG16_eval
if  [ ! -d $SAVE_PATH ];then
    mkdir -p $SAVE_PATH
fi

pushd $PROJ_DIR/models
# Eval model After Train based on last iteration
python SSD_VGG16_test.py --trained_model $PROJ_DIR/data/weights/gpu_checkpoints/ssd300_mAP_77.43_v2.pth \
                         --voc_root ${PYTORCH_TRAIN_DATASET}VOCdevkit \
                         --device mlu