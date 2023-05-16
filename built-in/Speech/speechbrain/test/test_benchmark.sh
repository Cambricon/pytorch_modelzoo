#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 net-precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             net: ecapa"
    echo "|             precision: fp32, amp"
    echo "|             device: mlu, gpu"
    echo "|             option: [ddp], [ci]"
    echo "|  eg.1. bash test_benchmark.sh ecapa-fp32-mlu"
    echo "|      which means running ecapa-tdnn net on single MLU card."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh ecapa-fp32-mlu-ddp"
    echo "|      which means running ecapa-tdnn net on 4 MLU cards."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}
while getopts 'h:' opt; do
   case "$opt" in
       h)  usage ; exit 1 ;;
       ?)  echo "unrecognized optional arg : "; $opt; usage; exit 1;;
   esac
done
## 加载参数配置
config=$1
source ${CUR_DIR}/params_config.sh
set_configs "$config"

pushd ${CUR_DIR}/../models
    pip install -r requirements.txt
    python setup.py install
popd
export DATASET_NAME="voxceleb"

ECAPA_DIR=${CUR_DIR}/../models/recipes/VoxCeleb/SpeakerRec/
if [ -d "${ECAPA_DIR}/results" ]; then
    rm -rf ${ECAPA_DIR}/results
fi
cp -r $PYTORCH_TRAIN_DATASET/voxceleb/voxceleb_wav/results ${ECAPA_DIR}
run_cmd="train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --device $device $train_params"
if [[ $ddp == "True" ]]; then
    run_cmd="-m torch.distributed.run --nproc_per_node=$nproc_per_node $run_cmd --distributed_launch --distributed_backend=$backend "
fi

echo python $run_cmd
pushd ${CUR_DIR}/../models/recipes/VoxCeleb/SpeakerRec/
eval "python $run_cmd "
popd
