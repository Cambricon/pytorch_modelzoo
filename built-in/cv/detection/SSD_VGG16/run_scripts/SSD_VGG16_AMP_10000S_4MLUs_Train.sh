# MLU Train Scripts
CUR_DIR=$(cd $(dirname $0);pwd)
export MLU_VISIBLE_DEVICES=0,1,2,3
ITERATIONS=10000
SAVE_PATH=$PROJ_DIR/data/output/train/MLU_SSD_VGG16_${ITERATIONS}S_AMP_4MLUs

if  [ ! -d $SAVE_PATH ];then
    mkdir -p $SAVE_PATH
fi

pushd $PROJ_DIR/models
# Training
python SSD_VGG16_train.py   --dataset_root ${PYTORCH_TRAIN_DATASET}VOCdevkit \
                            --save_folder $SAVE_PATH \
                            --dataset VOC \
                            --seed 42 \
                            --iters $ITERATIONS \
                            --device mlu \
                            --multiprocessing-distributed \
                            --world-size 1 \
                            --rank 0 \
                            --batch_size 64 \
                            --lr 2e-3 \
                            --dist-backend cncl \
                            --pyamp \
                            --pretrained_path ${PYTORCH_TRAIN_CHECKPOINT}ssd_vgg16/basenet/ \
                            --dist_url "tcp://127.0.0.10:28501"

# Evaluation model After Train based on last iteration
python SSD_VGG16_test.py --trained_model $SAVE_PATH/mlu_weights_ssd300_VOC_$ITERATIONS.pth \
                         --voc_root ${PYTORCH_TRAIN_DATASET}VOCdevkit \
                         --device mlu \
                         --pyamp
