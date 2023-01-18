#!/usr/bin/env bash
pushd ./utils/
CUDA_PATH=/usr/local/cuda/
python build.py build_ext --inplace --force
if (($?!=0)); then
    echo "install rfbnet-benchmark failed !!"
    exit
else 
    echo "install rfbnet-benchmark success !!"
fi
popd
