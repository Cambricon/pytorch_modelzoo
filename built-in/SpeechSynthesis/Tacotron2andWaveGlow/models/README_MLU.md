# Tacotron 2 And WaveGlow v1.10 For PyTorch

This repository provides a script and recipe to train the Tacotron 2 And WaveGlow v1.10 model to achieve state of the art accuracy, and is tested and maintained by Cambricon Pytorch.

## Model overview

The Tacotron 2 And WaveGlow v1.10 model is based on (https://github.com/NVIDIA/DeepLearningExamples/tree/46ff3707e04683e41b79af0f94a74e45f8016786/PyTorch/SpeechSynthesis/Tacotron2) repository.

This repository maintenance is performed on MLU devices for training and prediction.

## Quick Start Guide

1.Download and preprocess the dataset. Use the ./scripts/prepare_dataset.sh download script to automatically download and preprocess the training, validation and test datasets. 
    `bash scripts/prepare_dataset.sh`
Data is downloaded to the ./LJSpeech-1.1 directory (on the host).


2. preprocessed mel-spectrograms

use the ./scripts/prepare_mels.sh script:

     `bash scripts/prepare_mels.sh`

The preprocessed mel-spectrograms are stored in the ./LJSpeech-1.1/mels directory.

3. install requirements.

pip install -r requirements.txt

4. Start training.

To start WaveGlow training, run:
    
    `bash platform/MLU_tacotron2_{FP32, AMP}_{1, 4, 8}NMLU_train.sh`

To start WaveGlow training, run:

    `bash platform/MLU_waveglow_{FP32, AMP}_{1, 4, 8}NMLU_train.sh`
