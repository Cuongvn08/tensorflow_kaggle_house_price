# -*- coding: utf-8 -*-

import os
import shutil
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from config import *
from data import *
from model import Model


''' numeric features
['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold']
'''

''' categorical features
['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition']
'''
################################################################################
## HELPFUL FUNCTIONS
################################################################################
# get batches
def get_batches(data_len, batch_size):
    batch_starts = range(0, data_len, batch_size)
    batch_ends = [batch_start + batch_size for batch_start in batch_starts]
    return zip(batch_starts, batch_ends)

# get loss
def get_loss(logits, labels, method='L1', weights=None, l2_beta=0.0):
    # compute loss
    if method == 'L1':
        diff = tf.abs(logits - labels)
    else:
        diff = tf.square(logits - labels)/2

    loss = tf.reduce_mean(diff)

    # add L2 regularization to loss
    if weights is not None and l2_beta > 0.0:
        l2_regu = 0.0
        for key in weights:
            l2_regu += tf.nn.l2_loss(weights[key])
        loss = tf.add(loss, tf.multiply(l2_beta,l2_regu))

    return loss

# get optimizer
def get_optimizer(learning_rate, optimizer):
    if optimizer == eOptimizer.Adam:
        return tf.train.AdamOptimizer(learning_rate = learning_rate,
                                      beta1 = 0.9,
                                      beta2 = 0.999,
                                      epsilon = 1e-10,
                                      use_locking = False,
                                      name = 'Adam')
    elif optimizer == eOptimizer.GD:
        return tf.train.GradientDescentOptimizer(learning_rate = learning_rate,
                                                 use_locking = False,
                                                 name = 'GradientDescent')

    elif optimizer == eOptimizer.RMS:
        return tf.train.RMSPropOptimizer(learning_rate = learning_rate,
                                         decay = 0.9,
                                         momentum = 0.0,
                                         epsilon = 1e-10,
                                         use_locking = False,
                                         centered = False,
                                         name = 'RMSProp')
    else:
        assert 'optimizer error.'

def print_log(logger, str):
    print(str)
    logger.write(str + '\n')
    logger.flush()


################################################################################
## MAIN PROGRAM
################################################################################
# create log file
with tf.name_scope('logger'):
    log_path = 'result/log.txt'
    if os.path.exists(log_path):
        os.remove(log_path)
    os.makedirs(os.path.dirname(log_path))

    logger = open(log_path, 'w')
    logger.write('Hello world.\n\n')

# create session
with tf.name_scope('session'):
    sess = tf.Session()

# load config
with tf.name_scope('load_config'):
    cfig = get_config()

# load data
with tf.name_scope('load_data'):
    data = Data()
    data.read_train_data(cfig[eKey.train_path], cfig[eKey.encoder_path], cfig[eKey.train_ratio])

    train_data  = data.get_train_data()
    train_label = data.get_train_label()

    eval_data  = data.get_eval_data()
    eval_label = data.get_eval_label()

# create placeholder
with tf.name_scope('placeholder'):
    num_features = train_data.shape[1]

    data = tf.placeholder(cfig[eKey.data_dtype], shape=[None, num_features], name='data')
    label = tf.placeholder(cfig[eKey.label_dtype], shape=[None], name='label')

    tf.add_to_collection('data', data)
    tf.add_to_collection('label', label)

# create model
with tf.name_scope('model'):
    model = Model()

# get train opt
with tf.name_scope('train'):
    train_logit = model.logit(data, True, cfig[eKey.dropout], logger)
    train_cost = get_loss(train_logit, label, method='L2')
    train_opt = get_optimizer(cfig[eKey.learning_rate], cfig[eKey.optimizer]).minimize(train_cost)

    train_summary_list = []
    train_summary_list.append(tf.summary.scalar('train_cost', train_cost))
    train_summary_merge = tf.summary.merge(train_summary_list)

# get eval opt
with tf.name_scope('eval'):
    tf.get_variable_scope().reuse_variables()
    eval_logit = model.logit(data, False)
    eval_cost = get_loss(eval_logit, label, method='L1')

    pred = eval_logit

    eval_summary_list = []
    eval_summary_list.append(tf.summary.scalar('eval_cost', eval_cost))
    eval_summary_merge = tf.summary.merge(eval_summary_list)

    tf.add_to_collection('pred', pred)

# initialize variables
with tf.name_scope('initialize_variables'):
    sess.run(tf.global_variables_initializer())

# create summary
with tf.name_scope('summary'):
    if os.path.exists(cfig[eKey.log_dir]) is True:
        shutil.rmtree(cfig[eKey.log_dir])
    os.makedirs(cfig[eKey.log_dir])

    summary_writer = tf.summary.FileWriter(cfig[eKey.log_dir], sess.graph)

# create saver
with tf.name_scope('saver'):
    if os.path.exists(cfig[eKey.checkpoint_dir]) is True:
        shutil.rmtree(cfig[eKey.checkpoint_dir])
    os.makedirs(cfig[eKey.checkpoint_dir])

    saver = tf.train.Saver(max_to_keep = None)

# train
train_batches = get_batches(train_data.shape[0], cfig[eKey.batch_size])
for start, end in train_batches:
    str = '[train] train_batches: start={0}, end={1}'.format(start, end)
    print_log(logger, str)

logger.write('\n')
with tf.name_scope('train'):
    with tf.device('/cpu:%d' % 0):
    #with tf.device('/gpu:%d' % 0):
        for step in range(cfig[eKey.num_epoch]):
            # train
            start_time = time.time()
            train_fetches = [train_logit, train_cost, train_opt, train_summary_merge]

            train_costs = []
            train_batches = get_batches(train_data.shape[0], cfig[eKey.batch_size])
            for start, end in train_batches:
                feed_dict = {}
                feed_dict[data] = train_data[start:end]
                feed_dict[label] = train_label[start:end]
                [_, tCost, _, tSummary] = sess.run(train_fetches, feed_dict)
                train_costs.append(tCost)

            # eval
            if step % cfig[eKey.eval_step] == 0:
                eval_fetches = [eval_logit, eval_cost, pred, eval_summary_merge]

                feed_dict = {}
                feed_dict[data] = eval_data
                feed_dict[label] = eval_label
                [_, eCost, ePred, eSummary] = sess.run(eval_fetches, feed_dict)

                # log and print
                elapsed_time = time.time() - start_time
                date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                str = '[train] {0} step {1:04}: tCost = {2:0.5}; eCost = {3:0.5}; time = {4:0.5}(s);'.\
                    format(date_time, step, np.mean(train_costs), eCost, elapsed_time)
                print_log(logger,str)

                # save summaries
                summary_writer.add_summary(tSummary, step)
                summary_writer.add_summary(eSummary, step)
                summary_writer.flush()

                # save checkpoint
                saver.save(sess, cfig[eKey.checkpoint_dir] + 'chk_step_%d.ckpt' % step)


logger.write('The end.')
logger.close()
print('The end.')
