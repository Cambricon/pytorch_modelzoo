# -*- coding: utf-8 -*-

'''
Janurary 2018 by Wei Li
liweihfyz@sjtu.edu.cn
https://www.github.cim/leviswind/transformer-pytorch
'''

from __future__ import print_function
import argparse
import codecs
import os
import random

import numpy as np

import torch
from hyperparams import Hyperparams as hp
from data_load import TestDataSet, load_de_vocab, load_en_vocab
from nltk.translate.bleu_score import corpus_bleu
from AttModel import AttModel
from torch.autograd import Variable

try:
    import cnmix
except ImportError:
    print("train without cnmix")


def eval(args):
    # Load data

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    enc_voc = len(de2idx)
    dec_voc = len(en2idx)

    # load model
    model = AttModel(hp, enc_voc, dec_voc)


    source_test = args.dataset_path + hp.source_test
    target_test = args.dataset_path + hp.target_test
    test_dataset = TestDataSet(source_test, target_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False)

    if args.device == "MLU":
        model.to(ct.mlu_device())
    elif args.device == "GPU":
        model.cuda()

    state = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(state['model'])
    if args.device == "MLU"  and args.cnmix:
        model, _ = cnmix.initialize(model, None, opt_level = args.opt_level)
        if isinstance(state, dict) and 'cnmix' in state:
            cnmix.load_state_dict(state['cnmix'])

    print('Model Loaded.')
    model.eval()

    with codecs.open(args.log_path, 'a', 'utf-8') as fout:
        list_of_refs, hypotheses = [], []
        for i, (x, sources, targets) in enumerate(test_loader):
            if (i == args.iterations):
                break
            # Autoregressive inference
            if args.device == "GPU":
                x_ = x.long().cuda()
                preds_t = torch.LongTensor(np.zeros((x.size()[0], hp.maxlen), np.int32)).cuda()
                preds = Variable(preds_t).cuda()
            elif args.device == "MLU":
                x_ = x.long().to('mlu')
                preds_t = torch.LongTensor(np.zeros((x.size()[0], hp.maxlen), np.int32)).to('mlu')
                preds = Variable(preds_t.to('mlu'))
            else:
                x_ = x.long()
                preds_t = torch.LongTensor(np.zeros((x.size()[0], hp.maxlen), np.int32))
                preds = Variable(preds_t)

            for j in range(hp.maxlen):
                _, _preds, _ = model(x_, preds)
                preds_t[:, j] = _preds.data[:, j]
                preds = Variable(preds_t.long())
            preds = preds.data.cpu().numpy()

            # Write to file
            for source, target, pred in zip(sources, targets, preds):  # sentence-wise
                got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                fout.write("- source: " + source + "\n")
                fout.write("- expected: " + target + "\n")
                fout.write("- got: " + got + "\n\n")
                fout.flush()

                # bleu score
                ref = target.split()
                hypothesis = got.split()
                if len(ref) > 3 and len(hypothesis) > 3:
                    list_of_refs.append([ref])
                    hypotheses.append(hypothesis)

        # Calculate bleu score
        score = corpus_bleu(list_of_refs, hypotheses)
        fout.write("Bleu Score = " + str(100 * score))
        print("Bleu Score = {}".format(100 * score))
    if os.getenv('AVG_LOG'):
        with open(os.getenv('AVG_LOG'), 'a') as train_avg:
            train_avg.write('Bleu Score:{}\n'.format(100 * score))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer evaluation.")
    parser.add_argument('--device', default='MLU', type=str, help='set the type of hardware used for evaluation.')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--pretrained', default='model_epoch_20.pth', type=str, help='training ckps path')
    parser.add_argument('--batch-size', default=32, type=int, help='evaluation batch size.')
    parser.add_argument('--workers', default=4, type=int, help='number of workers.')
    parser.add_argument('--log-path', default='output.txt', type=str, help='evaluation file path.')
    parser.add_argument('--dataset-path', default='corpora/', type=str, help='The path of imagenet dataset.')
    parser.add_argument('--iterations', default=-1, type=int, help="Number of training iterations.")
    parser.add_argument('--bitwidth', default=8, type=int, help="Set the initial quantization width of network training.")
    parser.add_argument('--cnmix', action='store_true', default=False, help='use cnmix for mixed precision training')
    parser.add_argument('--opt_level', type=str, default="O0", help='choose level of mixing precision')
    args = parser.parse_args()

    if args.device == "MLU":
        import torch_mlu
        import torch_mlu.core.mlu_model as ct

    eval(args)



