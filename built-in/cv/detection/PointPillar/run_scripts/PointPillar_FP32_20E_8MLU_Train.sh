#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
TRANS_DIR=$(cd ${CUR_DIR}/../models/;pwd)

if [ -z $PYTORCH_TRAIN_DATASET ]; then
    echo "[ERROR] Please set PYTORCH_TRAIN_DATASET."
    exit 1
fi

pushd $TRANS_DIR
pip install -r requirements.txt
# 编译pcdet包
python setup.py develop
# 清除历史训练数据
if [ -d "output" ]; then
    rm -rf "output"
fi
# 准备数据集目录
if [ -d "data/nuscenes" ]; then
    rm data/nuscenes
fi
ln -sf "${PYTORCH_TRAIN_DATASET}/nuScenes" "./data/nuscenes"
pushd $TRANS_DIR/tools
export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash scripts/dist_train.sh 8 --cfg_file cfgs/nuscenes_models/cbgs_pp_multihead.yaml --fix_random_seed --device mlu --set OPTIMIZATION.BATCH_SIZE_PER_GPU 2 OPTIMIZATION.LR 0.003 OPTIMIZATION.LR_WARMUP True

popd
popd
