import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from utils.dsp import *
from utils import hparams as hp
from utils.paths import Paths
from pathlib import Path
from torch.utils.data.distributed import DistributedSampler


###################################################################################
# WaveRNN Dataset #################################################################
###################################################################################


class VocoderDataset(Dataset):
    def __init__(self, path: Path, dataset_ids, batch_size, seq_len):
        self.metadata = dataset_ids
        self.coarse_path = path/'coarse'
        self.fine_path = path/'fine'
        self.batch_size = batch_size
        self.seq_len = seq_len


    def __getitem__(self, index):
        item_id = self.metadata[index]
        coarse = np.load(self.coarse_path/f'{item_id}.npy').astype(np.float32)
        fine = np.load(self.fine_path/f'{item_id}.npy').astype(np.float32)
        wav_len = coarse.shape[1] // self.batch_size
        coarse = coarse[:, :wav_len * self.batch_size]
        fine = fine[:, :wav_len * self.batch_size]
        coarse = np.reshape(coarse, (self.batch_size, -1)).astype(np.float32)
        fine = np.reshape(fine, (self.batch_size, -1)).astype(np.float32)
        if wav_len - self.seq_len < 1:
            pad_zero = np.zeros((self.batch_size,self.seq_len + 2),dtype=np.float32)
            coarse = np.concatenate([coarse, pad_zero], axis=1)
            fine = np.concatenate([fine, pad_zero], axis=1)
        return coarse, fine

    def __len__(self):
        return len(self.metadata)


def get_deepmind_datasets(path: Path, batch_size, seq_len, distributed_run, num_workers):

    with open(path/'dataset_WaveRNN.pkl', 'rb') as f:
        dataset = pickle.load(f)
    dataset_ids = [x[0] for x in dataset]

    random.seed(1234)
    random.shuffle(dataset_ids)
    test_samples = int(len(dataset) * hp.dpm_test_rate)
    test_ids = dataset_ids[-test_samples:]
    train_ids = dataset_ids[:-test_samples]

    train_dataset = VocoderDataset(path, train_ids, batch_size, seq_len)
    test_dataset = VocoderDataset(path, test_ids, batch_size, seq_len)

    if distributed_run:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        shuffle = False
    else:
        test_sampler = None
        train_sampler = None
        shuffle = True

    train_set = DataLoader(train_dataset,
                           sampler=train_sampler,
                           shuffle=shuffle,
                           collate_fn=collate_vocoder,
                           batch_size=1,
                           num_workers=num_workers,
                           pin_memory=True)

    test_set = DataLoader(test_dataset,
                          sampler=test_sampler,
                          shuffle=False,
                          batch_size=1,
                          num_workers=4,
                          pin_memory=True)

    return train_set, test_set


def collate_vocoder(batch):
    coarse = torch.tensor(batch[0][0])
    fine = torch.tensor(batch[0][1])


    return coarse, fine


