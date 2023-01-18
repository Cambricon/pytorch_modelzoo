#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
PIX2PIX_DIR=$(cd ${CUR_DIR}/../models/;pwd)

pushd $PIX2PIX_DIR
BATCH_SIZE=1
python train.py --name facades_pix2pix_resnet_9blocks --model pix2pix --netG resnet_9blocks \
                --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --display_id 0  --seed 0 \
                --dataroot $FACADES_PATH --batch_size $BATCH_SIZE --device mlu
popd
