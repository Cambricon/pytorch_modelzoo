test -d "$MT5_CHECKPOINT_DIR" || { echo "error: MT5_CHECKPOINT_DIR does not exist, please source env.sh"; exit 1; }
test -f "$CSL_DIR/benchmark/ts/train.tsv" || { echo "error: CSL_DIR/benchmark/ts/train.tsv does not exist, please source env.sh"; exit 1; }
test -f "$CSL_DIR/benchmark/ts/dev.tsv" || { echo "error: CSL_DIR/benchmark/ts/dev.tsv does not exist, please source env.sh"; exit 1; }
test -n "$MT5_SAVED_MODEL_DIR" || { echo "error: MT5_SAVED_MODEL_DIR does not exist, please source env.sh"; exit 1; }
test -n "$MT5_SAVED_MODEL_NAME" || { echo "error: MT5_SAVED_MODEL_NAME does not exist, please source env.sh"; exit 1; }
test -n "$DATASET_NAME" || { echo "error: DATASET_NAME does not exist, please source env.sh"; exit 1; }