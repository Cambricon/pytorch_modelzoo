#/bin/bash
usage() {
    echo "Usage: "
    echo "-------------------------------------------------------------"
    echo "parameter1: 0)MLU, 1)GPU."
    echo "parameter2: 0)single card, 1)multiprocessing-distributed"
    echo "parameter3: 0)precheckin, 1)daily, 2)weekly, 3)benchmark"
    echo "parameter4: -1)no cnmix, 0)O0, 1)O1, 2)O2, 3)O3."
    echo "-------------------------------------------------------------"
}

# Checkout envs
if [ -z $IMAGENET_TRAIN_DATASET ]; then
  echo "[ERROR] Please set IMAGENET_TRAIN_DATASET."
  exit 1
fi

if [ -z $IMAGENET_TRAIN_CHECKPOINT ]; then
  echo "[ERROR] Please set IMAGENET_TRAIN_CHECKPOINT."
  exit 1
fi

batch_size="256"
num_workers="100"
mlu_model=`cat /proc/driver/*/*/*/information | grep "Device name" | uniq | awk -F ":" '{print $2}' | awk -F "-" '{print $1}'`
# Use MLU or GPU
ddp=""
if [[ $1 -eq 1 ]];then
    device="cuda"
    if [[ $2 -eq 1 ]];then
        ddp="--distributed --dist-backend nccl"
    fi
else
    device="mlu"
    if [[ $2 -eq 1 ]];then
        ddp="--distributed --dist-backend cncl"
    fi
fi

# Use cnmix
cnmix=""
if [[ $4 -ne -1 ]]; then
    cnmix_level="O${4}"
    cnmix="--cnmix --opt_level ${cnmix_level}"
fi

# Set for precheckin, daily, weekly, or benchmark
eval_iterations=""
if [ $3 -eq 0 ];then
  train_iterations=2
  eval_iterations="--eval_iterations 2"
elif [ $3 -eq 1 ];then
  train_iterations=1000
elif [ $3 -eq 3 ];then
  export MLU_ADAPTIVE_STRATEGY_COUNT=100
  train_iterations=`expr ${MLU_ADAPTIVE_STRATEGY_COUNT} + 200`
  if [ ${mlu_model} == "MLU370" ]; then
    export MLU_ADAPTIVE_STRATEGY_COUNT=50
    train_iterations=`expr ${MLU_ADAPTIVE_STRATEGY_COUNT} + 50`
    batch_size="512"
    if [[ $2 -eq 1 ]]; then
      num_workers="12"
      DEVICE_COUNT=`python -c 'import torch_mlu.core.mlu_model as ct;\
                      print(ct.device_count())'`
      if [[ ${DEVICE_COUNT} -eq 16 || ${DEVICE_COUNT} -eq 8 ]]; then
              num_workers="7"
      fi
    fi
  fi
else
  train_iterations=1000
fi

python src/train.py \
    --config configs/inception_v2/inception_v2_train_ddp_daily.yaml \
    --train-workers $num_workers \
    --train-batch-size $batch_size \
    --train-dataset $IMAGENET_TRAIN_DATASET/train \
    --resume $IMAGENET_TRAIN_CHECKPOINT/inception_v2/epoch_49.pth \
    --valid-dataset $IMAGENET_TRAIN_DATASET/val \
    --device $device \
    --train_iterations $train_iterations \
    $eval_iterations \
    $ddp    \
    $cnmix
