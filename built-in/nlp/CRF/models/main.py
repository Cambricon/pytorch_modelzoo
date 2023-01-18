import torch
import torchcrf
from nltk import data
import nltk
from torchcrf import CRF
import numpy as np
from sklearn.model_selection import train_test_split
import random
import argparse
import copy
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../../../../tools/utils/")
from metric import MetricCollector
parser = argparse.ArgumentParser(description='CRF')
parser.add_argument('--device', default='cpu', type=str,
                        help='Use cpu gpu or mlu device')
parser.add_argument('--data', default="./treebank/",
                        type=str, metavar='DIR', help='path to dataset')
parser.add_argument('--iters', type=int, default=50, metavar='N',
                        help='iters per epoch')
args = parser.parse_args()

seed = 66

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if args.device == "gpu":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if args.device == 'mlu':
    import torch_mlu
    import torch_mlu.core.mlu_model as ct

def count_word_freq(sents):
    word_tag_pair_freq = dict()
    word_freq = dict()
    for sent in sents:
        for pair in sent:
            if pair not in word_tag_pair_freq.keys():
                word_tag_pair_freq[pair] = 1
            else:
                word_tag_pair_freq[pair] = word_tag_pair_freq[pair] + 1
            word = pair[0]
            if word not in word_freq.keys():
                word_freq[word] = 1
            else:
                word_freq[word] = word_freq[word] + 1
    return word_freq, word_tag_pair_freq

def build_label_idx(label_set):
    i = 0
    label2idx = dict()
    idx2label = dict()
    for label in label_set:
        label2idx[label] = i
        idx2label[i] = label
        i += 1
    return label2idx, idx2label

def build_word_idx(word_set):
    i = 0
    word2idx = dict()
    idx2word = dict()
    for word in word_set:
        word2idx[word] = i
        idx2word[i] = word
        i += 1
    return word2idx, idx2word

def compute_prob(word_freq, word_tag_pair_freq):
    freq_table = np.zeros((num_word, num_tag))
    for tup in word_tag_pair_freq.keys():
        r, c = word2idx[tup[0]], label2idx[tup[1]]
        freq_table[r, c] = float(word_tag_pair_freq[tup] / word_freq[tup[0]])
    return freq_table

def build_sent_prob(prob_dict, sent):
    sent_prob = []
    sent_tags = np.zeros(max_len, dtype=np.long)
    for i, tup in enumerate(sent):
        sent_prob.append(prob_dict[word2idx[tup[0]], :])
        sent_tags[i] = label2idx[tup[1]]
    sent_lst = np.array(sent_prob)
    return sent_prob, sent_tags

def gen_mask(sents):
    mask = torch.zeros((len(sents), max_len), dtype=torch.bool)
    for i, sent in enumerate(sents):
        for j, tup in enumerate(sent):
            if tup[0] != pad_token:
                mask[i, j] = 1
    return mask

def per_word_err(predict, label):
    total = 0
    err = 0
    for i in range(len(predict)):
        for j in range(len(predict[i])):
            if label[i][j] != predict[i][j]:
                err += 1
            total += 1
    return err / total

def mse(a, b):
    epsilon = 1.0 / 16384
    nan_mask = a != a
    diff = a - b
    diff[nan_mask] = 0
    diff = diff.abs().pow(2).sum()
    a_pow_sum = a.pow(2).sum()
    if diff <= (2 * epsilon) * (2 * epsilon):
        diff = 0.0
    if a_pow_sum <= epsilon:
        a_pow_sum += epsilon
    diff = torch.div(diff, (a_pow_sum * 1.0))
    return diff.sqrt()

def cpu_model(model, ten_x_train, ten_y_train, mask_train, test_data, x_test, y_test):
    loss = model(ten_x_train, ten_y_train, mask=mask_train)
    loss.backward()
    mask_test = gen_mask(test_data)
    ten_x_test = torch.from_numpy(x_test)

    decode_seq = model.decode(ten_x_test, mask = mask_test)
    per_word_err_cpu = per_word_err(decode_seq, y_test)

    return loss, per_word_err_cpu

if __name__ == "__main__":
    data.path.append(args.data)
    sents = nltk.corpus.treebank.tagged_sents()
    print("==================data loading completed==================")
    max_len = 0
    for sent in sents:
        max_len = max(max_len, len(sent))
    pad_token = "<PAD>"
    pad_label = "PAD"
    padded_sents = []
    for i, sent in enumerate(sents):
        curr_len = len(sent)
        padded_sents.append(sent + [(pad_token, pad_label)] * (max_len - curr_len))
    sents = padded_sents

    for i in range(len(sents)):
        for j in range(len(sent)):
            if sents[i][j][0] != pad_token:
                sents[i][j] = (sents[i][j][0].lower(), sents[i][j][1])

    train_data, test_data = train_test_split(sents, shuffle = True, test_size = 0.2, random_state = 66)

    tag_set = set([tup[1] for sent in sents for tup in sent])
    num_tag = len(tag_set)

    word_set = set([tup[0] for sent in sents for tup in sent])
    num_word = len(word_set)


    word_freq, word_tag_pair_freq = count_word_freq(sents)
    label2idx, idx2label = build_label_idx(tag_set)
    word2idx, idx2word = build_label_idx(word_set)

    prob_dict = compute_prob(word_freq, word_tag_pair_freq)

    train_sent_lst = []
    tr_tags_lst = []
    test_sent_lst = []
    te_tags_lst = []
    count = 0
    for i, sent in enumerate(train_data):
        tr_tmp1, tr_tmp2 = build_sent_prob(prob_dict, sent)
        train_sent_lst.append(tr_tmp1)
        tr_tags_lst.append(tr_tmp2)

    for i, sent in enumerate(test_data):
        te_tmp1, te_tmp2 = build_sent_prob(prob_dict, sent)
        test_sent_lst.append(te_tmp1)
        te_tags_lst.append(te_tmp2)

    x_train = np.array(train_sent_lst)
    y_train = np.array(tr_tags_lst)
    x_test = np.array(test_sent_lst)
    y_test = np.array(te_tags_lst)
    
    mask_train = gen_mask(train_data)

    use_cuda = torch.cuda.is_available()

    ten_x_train, ten_y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)

    model = CRF(num_tag, batch_first=True)
    
    all_loss = []
    all_per_word_err = []
    avg_log_metric = MetricCollector(enable_only_avglog=True)
    metric_collector = MetricCollector(
        enable_only_benchmark=True,
        record_elapsed_time=True,
        record_hardware_time=True if args.device == 'mlu' else False)
    for i in range(args.iters):
        model.reset_parameters()
        cpu_loss, per_word_err_cpu = cpu_model(
                                              model.cpu(), ten_x_train.cpu(), 
                                              ten_y_train.cpu(), mask_train.cpu(),
                                              test_data, x_test, y_test)
        if use_cuda and args.device == "gpu":
            ten_y_train = ten_y_train.long().cuda()
            ten_x_train = ten_x_train.cuda()
            mask_train = mask_train.cuda()
            model = model.cuda()
        elif args.device == "mlu":
            # Only calculate the time of mlu，so place is here.
            metric_collector.place()
            ten_y_train = ten_y_train.to(ct.mlu_device(), non_blocking=True)
            ten_x_train = ten_x_train.to(ct.mlu_device(), dtype =torch.float, non_blocking=True)
            mask_train = mask_train.to(ct.mlu_device(), non_blocking=True)
            model = model.to(ct.mlu_device())

        loss = model(ten_x_train, ten_y_train, mask=mask_train)
        all_loss.append(loss.detach().item())

        mask_test = gen_mask(test_data)
        ten_x_test = torch.from_numpy(x_test)
        if use_cuda and args.device == "gpu":
            ten_x_test = ten_x_test.cuda()
            mask_test = mask_test.cuda()
        elif args.device == "mlu":
            ten_x_test = ten_x_test.to(ct.mlu_device(), dtype =torch.float, non_blocking=True)
            mask_test = mask_test.to(ct.mlu_device(), non_blocking=True)

        decode_seq = model.decode(ten_x_test, mask = mask_test)
        metric_collector.record()

        per_word_err_mlu = per_word_err(decode_seq, y_test)
        all_per_word_err.append(per_word_err_mlu)
        print("iter: {}   loss: {}  per-word-error: {}  loss mse: {}  per-word-error diff: {} ".format(
                                                      i, loss.detach().item(),
                                                      per_word_err(decode_seq, y_test),
                                                      mse(loss.cpu(), cpu_loss),
                                                      per_word_err_cpu-per_word_err_mlu))
    # insert metrics and dump metrics
    acc = [torch.mean(torch.Tensor(all_per_word_err)).item(),
           torch.std(torch.Tensor(all_per_word_err)).item()]
    metric_collector.insert_metrics(
        net = "CRF",
        batch_size = ten_x_train.shape[1],
        precision = "fp32",
        cards = 1,
        DPF_mode = "single")
    avg_log_metric.insert_metrics(
        net = "CRF",
        accuracy = acc)
    metric_collector.dump()
    avg_log_metric.dump()
    print("The avg of per-word-error is {}.".format(torch.mean(torch.Tensor(all_per_word_err))))
    print("The std of per-word-error is {}.".format(torch.std(torch.Tensor(all_per_word_err))))
