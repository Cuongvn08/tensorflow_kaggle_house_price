# -*- coding: utf-8 -*-

'''
https://github.com/dydokamil/kaggle-house-prices
'''

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import LabelBinarizer
import pickle


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

class Data():
    def __init__(self):
        self.train_data     = []
        self.train_label    = []

        self.eval_data      = []
        self.eval_label     = []

        self.test_data      = []
        self.test_label     = []

    def read_train_data(self, data_path, encoders_path=None, train_ratio=0.8):
        if train_ratio > 1 or train_ratio < 0:
            assert False, '[data] train_ratio is not valid !!!'

        # load data
        # must set keep_default_na False, otherwise it will keep the default NA
        # that causes the values of feature is not only string but also number
        df = pd.read_csv(data_path, keep_default_na=False)
        df = df.drop(['Id'], 1)

        encoded_data = []
        encoders = [] if not encoders_path else pickle.load(open(encoders_path, 'rb'))

        # visualize data
        #visualize_data(df)

        # encode data
        for feature in df:
            data = df[feature]

            encoder = None
            if df[feature].dtype == 'object': # categorical feature
                print(feature)
                encoder = LabelBinarizer()
                encoder.fit(list(set(df[feature])))
                data = encoder.transform(data)
            else: # numeric features
                df[feature] = df[feature].fillna(df[feature].mean())
                if skew(df[feature]) > 0.75:
                    df[feature] = np.log1p(df[feature])

            data = np.array(data, dtype=np.float32)
            encoded_data.append(data)
            encoders.append(encoder)
        pickle.dump(encoders, open('result/encoder/encoders.pickle', 'wb'))

        # load data
        data = encoded_data[:-1]
        label = encoded_data[-1]

        # split indices
        num_data = data.shape[0]
        num_train = np.int(num_data * train_ratio)
        num_eval = num_data - num_train

        train_indices = random.sample(range(num_data), num_train)
        eval_indices = [i for i in range(num_data) if i not in train_indices]

        # split train data
        for i in train_indices:
            self.train_data.append(data[i])
            self.train_label.append(label[i])
        self.train_data = np.asarray(self.train_data)
        self.train_label = np.asarray(self.train_label)

        print('[data] shape of train data = {0}'.format(self.train_data.shape))
        print('[data] shape of train label = {0}'.format(self.train_label.shape))

        # split eval data
        for i in eval_indices:
            self.eval_data.append(data[i])
            self.eval_label.append(label[i])
        self.eval_data = np.asarray(self.eval_data)
        self.eval_label = np.asarray(self.eval_label)

        print('[data] shape of eval data = {0}'.format(self.eval_data.shape))
        print('[data] shape of eval label = {0}'.format(self.eval_label.shape))

    def read_test_data(self, data_path):
        df = pd.read_csv(data_path)

        self.test_data = df.values[:,:].reshape([-1, 28, 28, 1])
        self.test_label = None

        print('[data] shape of test data = {0}'.format(self.test_data.shape))
        print('[data] shape of test label = None')

    def get_train_data(self):
        return self.train_data
    
    def get_train_label(self):
        return self.train_label
    
    def get_eval_data(self):
        return self.eval_data
    
    def get_eval_label(self):
        return self.eval_label
    
    def get_test_data(self):
        return self.test_data
    
    def get_test_label(self):
        return self.test_label

def visualize_data(df):
    ## label
    # compute the skewness of SalePrice
    print('[data] Skewness of SalePrices = {0}'.format(skew(df['SalePrice'])))

    # plot the distribution of SalePrice
    sale_price = pd.DataFrame({ "price"             : df['SalePrice'],
                                "normalized_price"  : np.log1p(df['SalePrice'])})
    sale_price.hist()
    plt.suptitle('skewness of SalePrice')

    ## numeric features
    # get all numeric features
    numeric_features = df.columns[df.dtypes != "object"]
    print('[train] Number of numeric features = {0}'.format(len(numeric_features)))
    print('[train] Numeric_features = {0}'.format(numeric_features))

    # compute skewness
    print('[data] Skewness of all numeric features:')
    for feature in numeric_features:
        print('Skewness of {0}: {1}'.format(feature, skew(df[feature])))

    # plot the distribution of SalePrice
    for feature in numeric_features:
        df_feature = pd.DataFrame({feature: df[feature],
                                   feature+"_log(price + 1)": np.log1p(df[feature])})
        df_feature.hist()
        plt.suptitle(feature)

    ## categorical features
    # get all categorical features
    cate_features = df.columns[df.dtypes == "object"]
    print('[train] number of categorical features = {0}'.format(len(cate_features)))
    print('[train] cate_features = {0}'.format(cate_features))

    plt.pause(0.1)
