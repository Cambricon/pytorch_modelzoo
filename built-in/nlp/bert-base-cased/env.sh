echo "Setting up envs..."
export DATASET_NAME="SQuADv1.1"
export CNCL_MLULINK_TIMEOUT_SECS=-1
if [ -z $SQUAD_DIR ]; then
  echo "please set environment variable SQUAD_DIR."
  exit 1
fi
echo "SQUAD_DIR is "$SQUAD_DIR
