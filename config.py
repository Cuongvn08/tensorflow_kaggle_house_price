# -*- coding: utf-8 -*-

import numpy as np
from enum import Enum


class eKey(Enum):
    train_path       = 'train_path'
    test_path        = 'data_path'

    log_dir         = 'log_dir'
    checkpoint_dir  = 'checkpoint_dir'
    output_dir      = 'output_dir'

    train_ratio     = 'train_ratio'
    batch_size      = 'batch_size'
    num_epoch       = 'num_epoch'
    eval_step       = 'eval_step'

    data_dtype      = 'data_dtype'
    label_dtype     = 'label_dtype'

    learning_rate   = 'learning_rate'
    dropout         = 'dropout'
    l2_beta         = 'l2_beta'


def get_config():
    config = {}

    config[eKey.train_path]    = 'data/train.csv'
    config[eKey.test_path]     = 'data/test.csv'

    config[eKey.log_dir]       = 'result/log/'
    config[eKey.checkpoint_dir]= 'result/checkpoint/'
    config[eKey.output_dir]    = 'result/output/'

    config[eKey.train_ratio]   = 0.9999
    config[eKey.batch_size]    = 100
    config[eKey.num_epoch]     = 10000
    config[eKey.eval_step]     = 5

    config[eKey.data_dtype]    = np.float32
    config[eKey.label_dtype]   = np.int64

    config[eKey.learning_rate] = 0.0001
    config[eKey.dropout]       = 0.75
    config[eKey.l2_beta]       = 0.0001

    return config
