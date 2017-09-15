# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew


### STEP1: settings
class settings(Enum):
    train_path    = 'data/train.csv'
    test_path     = 'data/test.csv'

    def __str__(self):
        return self.value
    
### STEP2: data processing
numeric_features = \
['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
'MoSold', 'YrSold']

categorical_features = \
['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
'SaleType', 'SaleCondition']

def display_outlier(pd, feature=None):
    if feature is not None:
        fig, ax = plt.subplots()
        ax.scatter(x = pd[feature], y = pd['SalePrice'])
        plt.ylabel('SalePrice', fontsize=13)
        plt.xlabel(feature, fontsize=13)
        plt.show()      
    else:
        for feature in numeric_features:
            fig, ax = plt.subplots()
            ax.scatter(x = pd[feature], y = pd['SalePrice'])
            plt.ylabel('SalePrice', fontsize=13)
            plt.xlabel(feature, fontsize=13)
            plt.show()
        
def display_distrib(pd, feature=None):
    if feature is not None:
        plt.figure()
        sns.distplot(pd[feature] , fit=norm);
        (mu, sigma) = norm.fit(pd[feature])    
        
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
        plt.ylabel('Frequency')
        plt.title('SalePrice distribution')
        plt.show()
    else:
        for feature in numeric_features:
            plt.figure()
            sns.distplot(pd[feature] , fit=norm);
            (mu, sigma) = norm.fit(pd[feature])    
            
            plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
            plt.ylabel('Frequency')
            plt.title('SalePrice distribution')
            plt.show()
        
def data_processing(train_path, test_path):
    print('[data_processing] ', train_path)
    print('[data_processing] ', test_path)
    
    # load data
    train = pd.read_csv(str(train_path))
    test = pd.read_csv(str(test_path))
        
    #print('[data_processing] ', train.head(5))
    #print('[data_processing] ', test.head(5))
    
    # drop ID feature
    print('[data_processing] ', 'The train data size before dropping Id: {} '.format(train.shape))
    print('[data_processing] ', 'The test data size before dropping Id: {} '.format(test.shape))
    
    train.drop('Id', axis = 1, inplace = True)
    test.drop('Id', axis = 1, inplace = True)
    
    print('[data_processing] ', 'The train data size after dropping Id: {} '.format(train.shape))
    print('[data_processing] ', 'The test data size after dropping Id: {} '.format(test.shape))    
    
    # analyze and remove huge outliers: GrLivArea, ...
    display_outlier(train, 'GrLivArea')
    train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
    display_outlier(train, 'GrLivArea')
      
    # analyze and normalize distribution of the target feature (SalePrice)
    display_distrib(train, 'SalePrice')
    train["SalePrice"] = np.log1p(train["SalePrice"])
    display_distrib(train, 'SalePrice')
    
    # concatenate the train and test data
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    print('[data_processing] ', 'all_data size is : {}'.format(all_data.shape))    
    
    # fill missing data
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    print('[data_processing] ', missing_data)

    all_data["PoolQC"] = all_data["PoolQC"].fillna("None") #NA means "No Pool"
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None") #NA means "no misc feature"
    all_data["Alley"] = all_data["Alley"].fillna("None") #NA means "no alley access"
    all_data["Fence"] = all_data["Fence"].fillna("None") #NA means "no fence"
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None") #NA means "no fireplace"
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].\
                                transform(lambda x: x.fillna(x.median())) # fill by the
                                # median LotFrontage of all neighborhood because they
                                # have same lot frontage
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna('None') #NaN means that there is no basement
    
    
    
    
    
    
    
### STEP3: model
def model():
    pass

### STEP 4: loss optimization
def loss_optimization():
    pass
    
### STEP 5: analysis
def analysis():
    pass

### MAIN
data_processing(settings.train_path, settings.test_path)

print('[main] ', 'The end!')
