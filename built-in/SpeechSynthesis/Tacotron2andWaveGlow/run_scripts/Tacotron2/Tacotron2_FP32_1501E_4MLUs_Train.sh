
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
WORK_DIR=$(cd ${CUR_DIR}/../../models/;pwd)

pushd $WORK_DIR

export MLU_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 train.py -m Tacotron2 -o output/ -lr 1e-3 --epochs 1501 -bs 48 --weight-decay 1e-6 --grad-clip-thresh 1.0 --log-file nvlog.json --anneal-steps 500 1000 1500 --anneal-factor 0.1 -d ${PYTORCH_TRAIN_DATASET}/TTS/ --use-mlu --dist-backend cncl

popd