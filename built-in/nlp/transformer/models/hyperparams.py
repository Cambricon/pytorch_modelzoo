# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = 'train.tags.de-en.de'
    target_train = 'train.tags.de-en.en'
    source_test = 'IWSLT16.TED.tst2014.de-en.de.xml'
    target_test = 'IWSLT16.TED.tst2014.de-en.en.xml'


    # training
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logs' # log directory

    model_dir = 'ckps'  # saving directory

    # model
    maxlen = 10 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 20 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    base_dropout = False
    sinusoid = True # If True, use sinusoid. If false, positional embedding.
    eval_epoch = 20  # epoch of model for eval
    preload = None  # epcho of preloaded model for resuming training

