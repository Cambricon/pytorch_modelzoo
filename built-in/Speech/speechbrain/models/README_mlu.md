# SpeechBrain on MLU:

Train with MLU backend is available.
Float32 training is supported.

AMP is not supported.

## Supported networks

- ecapa tdnn

## Install

```
pip install -r requirements.txt
python setup.py install
```

## Training conformer

At first, pre-trained ckpt needs to be copied to
```
recipes/LibriSpeech/ASR/transformer/results/transformer/{seed}/save 
```

Or download it from hugging-face (see README.md)

```
cd recipes/LibriSpeech/ASR/transformer/
python train.py hparams/conformer_small.yaml --device mlu # (Single card training)
bash scratch_8mlu.sh  #(8-cards multi-card training)
```

## Test conformer

At the end of training, SpeechBrain will run into eval mode as default.

For network test, copy a trained checkpoint to results/transformer/{seed}/save can resume it.

Using speechbrain/cambricon/test_benchmark.sh, we can test network with specified params:

```
bash test_benchmark.sh mlu          # Run MLU train and test
export MLU_VISIBLE_DEVICES=0,1,2,3  # Using 4 cards
bash test_benchmark.sh mlu-ddp      # Run MLU distributed data parallel train and test
```

# Training Speaker verification using ECAPA-TDNN embeddings
Run the following command to train speaker embeddings using [ECAPA-TDNN](https://arxiv.org/abs/2005.07143):

`python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml`  (Single MLU card training)

```
cd cambricon
bash scratch_ecapa_8mlu.sh  #(8-cards multi-card training)
```

## Test Speaker verification
After training the speaker embeddings, it is possible to perform speaker verification using cosine similarity.  You can run it with the following command:

`python speaker_verification_cosine.py hparams/verification_ecapa.yaml`

```
bash test_benchmark.sh mlu          # Run MLU train and test
export MLU_VISIBLE_DEVICES=0,1,2,3  # Using 4 cards
bash test_benchmark.sh mlu-ddp      # Run MLU distributed data parallel train and test
```

## Modifications

To support MLU devices and CUDA devices at same time, speechbrain/speechbrain/core.py added
new params:

- --one_epoch     # Run only one epoch from start or from checkpoints
- --train_batches # Run only few batches for training. Used to debug or test.
- --valid_batches # Run only few batches for valid after each epoch.
- --eval_batches  # Run only few batches for eval after every 10 epochs or at the end.

And --device mlu is supoorted like --device cuda (default) or --device cpu.

## Note
1. Due to the problem of dataset preprocessing, we provide the preprocessed dataset for training the ECAPA-TDNN. Path of the preprocessed dataset is:
```
$PYTORCH_TRAIN_DATASET/voxceleb/voxceleb_wav/results
```
2. When trainning the ECAPA-TDNN with GPU,  please confirm the version of pytorch, you can install pytorch with the following commands:
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
