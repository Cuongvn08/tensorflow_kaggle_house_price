# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p


### STEP1: settings
class settings(Enum):
    train_path    = 'data/train.csv'
    test_path     = 'data/test.csv'

    def __str__(self):
        return self.value
    
### STEP2: data processing
'''
numeric_features = \
['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
'MoSold', 'YrSold']
'''

'''
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
'''

def display_outlier(pd, feature=None):
    if feature is not None:
        fig, ax = plt.subplots()
        ax.scatter(x = pd[feature], y = pd['SalePrice'])
        plt.ylabel('SalePrice', fontsize=13)
        plt.xlabel(feature, fontsize=13)
        plt.show()      
    else:
        #for feature in numeric_features:
        for feature in pd:
            if pd[feature].dtypes != "object":
                fig, ax = plt.subplots()
                ax.scatter(x = pd[feature], y = pd['SalePrice'])
                plt.ylabel('SalePrice', fontsize=13)
                plt.xlabel(feature, fontsize=13)
                plt.show()
        
def display_distrib(pd, feature):
    plt.figure()
    sns.distplot(pd[feature].dropna() , fit=norm);
    (mu, sigma) = norm.fit(pd[feature].dropna())    
    
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')
    plt.show()
        
def normalize_distrib(pd):
    for feature in pd:
        if pd[feature].dtype != "object":
            display_distrib(pd, feature)
            pd[feature] = np.log1p(pd[feature])
            display_distrib(pd, feature)
                
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

    all_data["PoolQC"] = all_data["PoolQC"].fillna("None") #data description says NA means "No Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general.
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None") #data description says NA means "no misc feature"
    all_data["Alley"] = all_data["Alley"].fillna("None") #data description says NA means "no alley access"
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
    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None") #NA means no masonry veneer
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0) #NA means no masonry veneer
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    all_data = all_data.drop(['Utilities'], axis=1) #For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it
    all_data["Functional"] = all_data["Functional"].fillna("Typ") #data description says NA means typical
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0]) #It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0]) #Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0]) #Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0]) #Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0]) #Fill in again with most frequent which is "WD"
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None") #Na most likely means No building class. We can replace missing values with None
    
    # transform some numeric features into categorical features
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)
    
    # do label encoding for categorical features
    categorical_features = \
    ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
     'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
     'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
     'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
     'YrSold', 'MoSold')
    for c in categorical_features:
        lbl = LabelEncoder() 
        lbl.fit(list(all_data[c].values))
        all_data[c] = lbl.transform(list(all_data[c].values))
    print('[data_processing] ', 'Shape all_data: {}'.format(all_data.shape))

    # add important features more
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF'] #feature which is the total area of basement, first and second floor areas of each house

    # normalize skewed features
    '''
    display_distrib(train)
    
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness = skewness[abs(skewness) > 0.75]
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        #all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)


    
    #display_distrib(train)
    '''

    # test
    for feature in all_data:
        print(feature, all_data[feature].dtype)
    
    
    
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
