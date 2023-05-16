CUR_DIR=$(cd $(dirname $0);pwd)
MODEL_DIR=$(cd ${CUR_DIR}/../models/;pwd)
source $CUR_DIR/../check_env.sh || {  exit 1; }
export MLU_VISIBLE_DEVICES=3

pushd $MODEL_DIR
python finetune.py \
    --pretrain_model=${MT5_CHECKPOINT_DIR} \
    --dev_data=${CSL_DIR}/benchmark/ts/dev.tsv \
    --saved_model_dir ${MT5_SAVED_MODEL_DIR} \
    --saved_model_name ${MT5_SAVED_MODEL_NAME} \
    --num_device 1 \
    --device MLU \
    --batch_size 4 \
    --mode evaluation 
popd