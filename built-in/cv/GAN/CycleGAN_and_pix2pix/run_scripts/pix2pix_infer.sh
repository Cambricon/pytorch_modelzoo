#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
PIX2PIX_DIR=$(cd ${CUR_DIR}/../models/;pwd)

pushd $PIX2PIX_DIR
python test.py --name facades_pix2pix_resnet_9blocks --model pix2pix --netG resnet_9blocks \
              --direction BtoA --dataset_mode aligned --norm batch --eval \
              --dataroot $FACADES_PATH --device mlu
popd
