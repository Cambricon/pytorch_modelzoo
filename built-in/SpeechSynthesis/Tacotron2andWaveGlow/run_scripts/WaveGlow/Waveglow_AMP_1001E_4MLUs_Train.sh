set -e

CUR_DIR=$(cd $(dirname $0);pwd)
WORK_DIR=$(cd ${CUR_DIR}/../../models/;pwd)

pushd $WORK_DIR
export MLU_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch --nproc_per_node=4 train.py -m WaveGlow -o output/ -lr 1e-4 --epochs 1001 -bs 4 --segment-length 8000 --weight-decay 0 --grad-clip-thresh 65504.0 --cudnn-benchmark -d ${PYTORCH_TRAIN_DATASET}/TTS/ --log-file nvlog.json --use-mlu --dist-backend cncl --pyamp $@popd
