CUR_DIR=$(cd $(dirname $0);pwd)
pushd ${CUR_DIR}/../models
export MLU_VISIBLE_DEVICES=0,1,2,3
python run_squad.py \
       --model_type bert \
       --model_name_or_path bert-base-cased \
       --do_eval --predict_file $SQUAD_DIR/dev-v1.1.json \
       --per_gpu_eval_batch_size 16 \
       --max_seq_length 384 \
       --doc_stride 128 \
       --output_dir bert_base_cased_ddp_from_scratch \
       --device_param mlu \
       --overwrite_output_dir
popd
