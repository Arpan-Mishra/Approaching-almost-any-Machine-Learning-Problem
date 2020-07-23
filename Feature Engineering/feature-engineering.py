import pandas as pd

# date time features

s = pd.date_range('2020-01-06','2020-01-10',freq = '10H').to_series()
print(s.head())

features = {

'dayofweek': s.dt.dayofweek.values,
'dayofyear': s.dt.dayofyear.values,
'hour': s.dt.hour.values,
'is_leap_year': s.dt.is_leap_year.values,
'quarter': s.dt.quarter.values,
'weekofyear': s.dt.weekofyear.values
}

print(features)
# aggregated features

def generate_features(df):
    # create features based on date

    df.loc[:,'year'] = df['date'].dt.year
    df.loc[:,'weekofyear'] = df['date'].dt.weekofyear
    df.loc[:,'month'] = df['month'].dt.month
    df.loc[:,'dayofweek'] = df['dayofweek'].dt.dayofweek
    df.loc[:,'weekend'] = (df['date'].dt.weekday>=5).astype(int)

    # aggregation dictionary

    aggs = {}

    # number of months the customer is active and the month he is most active on average
    aggs['month'] = ['nunique','mean']
    # same thing for the day
    aggs['weekofyear'] = ['nunique','mean']

    # for numeric columns num1 we calculate mean,sum,min,max per customer
    aggs['num1'] = ['mean','sum','min','max']

    # total count of the customer
    aggs['customer_id'] = ['size']

    # count of category 1 for a customer
    aggs['cat1'] = ['size']

    agg_df = df.groupby('customer_id').agg(aggs).reset_index()
    return agg_df


# we can then merge these features into our df using customer_id as the key 
# or we could just use it for analysis.
    
# In case of list of values, not individual values



import numpy as np
x = np.random.rand(10000,1)

feature_dict = {}

feature_dict['mean'] = np.mean(x)
feature_dict['var'] = np.var(x)
feature_dict['min'] = np.min(x)
feature_dict['max'] = np.max(x)
feature_dict['ptp'] = np.ptp(x)

# percentile feats
feature_dict['percentile_10'] = np.percentile(x,10)
feature_dict['percentile_60'] = np.percentile(x,60)
feature_dict['percentile_90'] = np.percentile(x,90)

# quantile feat
feature_dict['quantile_1'] = np.percentie(x,25)
feature_dict['quantile_2'] = np.percentie(x,50)
feature_dict['quantile_3'] = np.percentie(x,75)
feature_dict['quantile_4'] = np.percentie(x,99)

# ts fresh can be used for time series features when we have a list of features 
# in a given period of time
from tsfresh.feature_extraction import feature_calculators as fc

feature_dict['abs_energey'] = fc.abs_energy(x)
feature_dict['count_above_mean'] = fc.count_above_mean(x)
feature_dict['count_below_mean'] = fc.count_below_mean(x)
feature_dict['mean_abs_change'] = fc.mean_abs_change(x)
feature_dict['mean_change'] = fc.mean_change(x)

# polynomial features
import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.DataFrame(np.random.rand(100,2),columns = ['f1','f2'])
pf = preprocessing.PolynomialFeatures(degree = 2, 
interaction_only = False,
include_bias = False)
pf.fit(df)

poly_ft = pf.transform(df)

poly_df = pd.DataFrame(poly_ft,columns = [f'f{i}' for i in range(1,poly_ft.shape[1]+1)])

# binning
# 10 bins
df['bin_10'] = pd.cut(df['f1'],bins = 10,labels = False)
# 100 bins
df['bin_100'] = pd.cut(df['f1'],bins = 100,labels = False)

# log transformation - creates a low variance feature from high variance feature
df['f3'] = np.random.randn( 100,1 )

df['log_f3'] = df['f3'].apply(lambda x: np.log(1+x))

# can use exp faetures as well


# Treating missing values
# 1. for categorical we can simply treat nan has a separate catefory
# 2. for numeric: - mean,median,KNN 
from sklearn import impute
import numpy as np

x = np.random.randint(1,15,(10,6)).astype(float)
x.ravel()[np.random.choice(x.size,10,replace = False)] = np.nan
print(x)
# imputing using KNN (2 neighbours)
knn_imputer = impute.KNNImputer(n_neighbors = 2)
x = knn_imputer.fit_transform(x)
print(x)



