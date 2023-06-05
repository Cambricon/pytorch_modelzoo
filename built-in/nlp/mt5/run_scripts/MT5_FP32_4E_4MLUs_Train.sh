CUR_DIR=$(cd $(dirname $0);pwd)
MODEL_DIR=$(cd ${CUR_DIR}/../models/;pwd)
source $CUR_DIR/../check_env.sh || {  exit 1; }

pushd $MODEL_DIR
python finetune.py \
    --pretrain_model=${MT5_CHECKPOINT_DIR} \
    --train_data=${CSL_DIR}/benchmark/ts/train.tsv \
    --dev_data=${CSL_DIR}/benchmark/ts/dev.tsv \
    --saved_model_dir ${MT5_SAVED_MODEL_DIR} \
    --saved_model_name ${MT5_SAVED_MODEL_NAME} \
    --lr 0.0002 \
    --num_device 4 \
    --batch_size 4 \
    --num_epoch 4 \
    --device MLU \
    --distributed 
cd -