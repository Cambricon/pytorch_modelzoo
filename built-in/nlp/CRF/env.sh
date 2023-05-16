#bin/bash
#set -e
echo "Setting up envs..."

export DATASET_NAME="PTB"
export CRF_DATASET=$PWD/data/nltk_data
if [ ! -d $CRF_DATASET/treebank ]; then
    echo "dataset does not exists! Start downloading from remote....."
    mkdir -p $CRF_DATASET/treebank
    python models/download_dataset.py --data $CRF_DATASET/treebank
fi
echo "CRF_DATASET is "$CRF_DATASET
echo "If you have downloaded the dataset, you need to “export CRF_DATASET=YOUR DATA PATH” "

