# export CUDA_VISIBLE_DEVICES='0,1,2,3'
# export DATASET=/algo/modelzoo/datasets/imagenet/jpegs

python src/train_fix.py \
    --config configs/inception_v2/inception_v2_train_fix_ddp.yaml \
    --distributed \
    --train-dataset $DATASET/train \
    --valid-dataset $DATASET/val \
    --dist-backend cncl

