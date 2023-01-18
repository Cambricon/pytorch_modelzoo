"""Data loader"""

import random
import numpy as np
import os
import sys

import torch
from pytorch_pretrained_bert import BertTokenizer

import utils


class DataSet(torch.utils.data.Dataset):
    def __init__(self, data_dir, bert_model_dir, params, token_pad_idx=0):
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.seed = params.seed
        self.token_pad_idx = 0

        tags = self.load_tags()
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(tags)}
        params.tag2idx = self.tag2idx
        params.idx2tag = self.idx2tag
        self.tag_pad_idx = self.tag2idx['O']

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)

    def load_tags(self):
        tags = []
        file_path = os.path.join(self.data_dir, 'tags.txt')
        with open(file_path, 'r', encoding='utf-8') as file:
            for tag in file:
                tags.append(tag.strip())
        return tags

    def load_sentences_tags(self, sentences_file, tags_file, d):
        """Loads sentences and tags from their corresponding files. 
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        sentences = []
        tags = []

        with open(sentences_file, 'r', encoding='utf-8') as file:
            for line in file:
                # replace each token by its index
                tokens = self.tokenizer.tokenize(line.strip())
                sentences.append(self.tokenizer.convert_tokens_to_ids(tokens))
        
        with open(tags_file, 'r', encoding='utf-8') as file:
            for line in file:
                # replace each tag by its index
                tag_seq = [self.tag2idx.get(tag) for tag in line.strip().split(' ')]
                tags.append(tag_seq)

        # checks to ensure there is a tag for each token
        assert len(sentences) == len(tags)
        for i in range(len(sentences)):
            assert len(tags[i]) == len(sentences[i])

        # storing sentences and tags in dict d
        d['data'] = sentences
        d['tags'] = tags
        d['size'] = len(sentences)
    
    def load_data(self, data_type):
        """Loads the data for each type in types from data_dir.

        Args:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Returns:
            data: (dict) contains the data with tags for each type in types.
        """
        data = {}
        
        if data_type in ['train', 'val', 'test']:
            sentences_file = os.path.join(self.data_dir, data_type, 'sentences.txt')
            tags_path = os.path.join(self.data_dir, data_type, 'tags.txt')
            self.load_sentences_tags(sentences_file, tags_path, data)
        else:
            raise ValueError("data type not in ['train', 'val', 'test']")
        return data


class TrainDataSet(DataSet):
    def __init__(self, *args, **kwargs):
        super(TrainDataSet, self).__init__(*args, **kwargs)
        self.data = self.load_data('train')
    
    def __getitem__(self, idx):
        sentences = self.data['data'][idx]
        tags = self.data['tags'][idx]
         
        data_s = self.token_pad_idx * np.ones((self.max_len))
        data_t = self.tag_pad_idx * np.ones((self.max_len))
        len_s = min(self.max_len, len(sentences))
        len_t= min(self.max_len, len(tags))
        data_s[:len_s] = sentences[:len_s]
        data_t[:len_t] = tags[:len_t]
       
        return torch.LongTensor(data_s), torch.LongTensor(data_t) 

    def __len__(self):
        return self.data['size']

class ValDataSet(DataSet):
    def __init__(self, *args, **kwargs):
        super(ValDataSet, self).__init__(*args, **kwargs)
        self.data = self.load_data('val')
    
    def __getitem__(self, idx):
        sentences = self.data['data'][idx]
        tags = self.data['tags'][idx]
         
        data_s = self.token_pad_idx * np.ones((self.max_len))
        data_t = self.tag_pad_idx * np.ones((self.max_len))
        len_s = min(self.max_len, len(sentences))
        len_t= min(self.max_len, len(tags))
        data_s[:len_s] = sentences[:len_s]
        data_t[:len_t] = tags[:len_t]
       
        return torch.LongTensor(data_s), torch.LongTensor(data_t) 

    def __len__(self):
        return self.data['size']

class TestDataSet(DataSet):
    def __init__(self, *args, **kwargs):
        super(TestDataSet, self).__init__(*args, **kwargs)
        self.data = self.load_data('test')
    
    def __getitem__(self, idx):
        sentences = self.data['data'][idx]
        tags = self.data['tags'][idx]
         
        data_s = self.token_pad_idx * np.ones((self.max_len))
        data_t = self.tag_pad_idx * np.ones((self.max_len))
        len_s = min(self.max_len, len(sentences))
        len_t= min(self.max_len, len(tags))
        data_s[:len_s] = sentences[:len_s]
        data_t[:len_t] = tags[:len_t]
       
        return torch.LongTensor(data_s), torch.LongTensor(data_t) 

    def __len__(self):
        return self.data['size']
