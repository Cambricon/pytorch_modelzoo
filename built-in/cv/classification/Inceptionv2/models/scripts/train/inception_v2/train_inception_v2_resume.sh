# export CUDA_VISIBLE_DEVICES='0,1,2,3'
# export DATASET=/algo/modelzoo/datasets/imagenet/jpegs

python src/train.py \
    --config configs/inception_v2/inception_v2_train_ddp.yaml \
    --distributed \
    --train-dataset $DATASET/train \
    --valid-dataset $DATASET/val \
    --resume /algo/algo/zhaozhipeng/workspace/trainingcodezoo/Pytorch/Classification/inception_v2_train/ckps_inception_v2/last.pth \
    --dist-backend cncl
