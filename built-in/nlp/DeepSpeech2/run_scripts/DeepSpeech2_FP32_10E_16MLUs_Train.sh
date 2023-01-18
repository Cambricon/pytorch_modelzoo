CUR_DIR=$(cd $(dirname $0);pwd)
if [ ! -d "../models/LibriSpeech_dataset" ]; then
  ln -s $PYTORCH_TRAIN_DATASET/LibriSpeech_dataset  ../models
fi
pushd ${CUR_DIR}/../models/pytorch
export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
python -m torch.distributed.launch \
       --nproc_per_node=16 \
       train.py\
       --device  mlu \
       --acc 0.0 \
       --save_folder  ./models \
       --model_path  ./models/deepspeech2_final.pth.tar \
       --num_workers 32
popd
