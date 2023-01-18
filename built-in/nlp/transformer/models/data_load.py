# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp

import numpy as np
import codecs
import regex
import random
import torch
import torch.utils.data as data

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def load_de_vocab():
    vocab = [line.split()[0] \
             for line in codecs.open(
                     './data/IWSLT/preprocessed/de.vocab.tsv',
                     'r', 'utf-8').read().splitlines()\
             if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_en_vocab():
    vocab = [line.split()[0] \
             for line in codecs.open(
                     './data/IWSLT/preprocessed/en.vocab.tsv',
                     'r', 'utf-8').read().splitlines()\
             if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(source_sents, target_sents):
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [de2idx.get(word, 1) \
             for word in (source_sent + u" </S>").split()]
        y = [en2idx.get(word, 1) \
             for word in (target_sent + u" </S>").split()]
        if max(len(x), len(y)) <=hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    # Pad
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)],
                          'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)],
                          'constant', constant_values=(0, 0))

    return X, Y, Sources, Targets


class TrainDataSet(data.Dataset):
    def __init__(self, source_path, target_path):
        self.source_path = source_path
        self.target_path = target_path
        # In python2 codecs cannot handle special German characters,
        # so python2 and python3 have different results.
        # eg. Lebensr\xe4umen(python2) Lebensräumen(python3)
        de_sents = [regex.sub("[^\s\p{Latin}']", "", line) \
                    for line in codecs.open(
                            self.source_path,
                            'r',
                            'utf-8').read().split("\n") \
                    if line and line[0] != "<"]
        en_sents = [regex.sub("[^\s\p{Latin}']", "", line) \
                    for line in codecs.open(
                            self.target_path,
                            'r',
                            'utf-8').read().split("\n") \
                    if line and line[0] != "<"]
        self.data, self.label, _, __ = create_data(de_sents, en_sents)

    def __getitem__(self, index):
        return torch.LongTensor(self.data[index]), torch.LongTensor(self.label[index])

    def __len__(self):
        return len(self.data)

class TestDataSet(data.Dataset):
    def __init__(self, source_path, target_path):
        self.source_path = source_path
        self.target_path = target_path
        def _refine(line):
            line = regex.sub("<[^>]+>", "", line)
            line = regex.sub("[^\s\p{Latin}']", "", line)
            return line.strip()

        de_sents = [_refine(line) \
                    for line in codecs.open(
                            self.source_path,
                            'r', 'utf-8').read().split("\n") \
                    if line and line[:4] == "<seg"]
        en_sents = [_refine(line) \
                    for line in codecs.open(
                            self.target_path,
                            'r', 'utf-8').read().split("\n") \
                    if line and line[:4] == "<seg"]

        self.data, _, self.sources, self.targets = create_data(de_sents, en_sents)

    def __getitem__(self, index):
        return self.data[index], self.sources[index], self.targets[index]

    def __len__(self):
        return len(self.data)


