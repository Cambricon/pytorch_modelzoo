#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)

ecapa_tdnn_path=${CUR_DIR}/../models/recipes/VoxCeleb/SpeakerRec

pushd $ecapa_tdnn_path
  if [ ! -d "./results" ]; then
      cp -r ${PYTORCH_TRAIN_DATASET}/voxceleb/voxceleb_wav/results .
  fi
  python -m torch.distributed.run --nproc_per_node=8 train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --distributed_launch --distributed_backend cncl
popd
