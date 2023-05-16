#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)

ecapa_tdnn_path=${CUR_DIR}/../models/recipes/VoxCeleb/SpeakerRec

pushd $ecapa_tdnn_path
  if [ ! -d "./results" ]; then
      cp -r ${PYTORCH_TRAIN_DATASET}/voxceleb/voxceleb_wav/results .
  fi
  python speaker_verification_cosine.py hparams/verification_ecapa.yaml --distributed_launch --distributed_backend cncl
popd
