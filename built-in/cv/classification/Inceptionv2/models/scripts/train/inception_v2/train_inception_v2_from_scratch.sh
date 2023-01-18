#/bin/bash
export MLU_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
#export DATASET=/data/pytorch/datasets/imagenet_training

python src/train.py \
    --config configs/inception_v2/inception_v2_train_ddp.yaml \
    --distributed \
    --train-dataset $IMAGENET_TRAIN_DATASET/train \
    --valid-dataset $IMAGENET_TRAIN_DATASET/val \
    --device mlu \
    --dist-backend cncl
