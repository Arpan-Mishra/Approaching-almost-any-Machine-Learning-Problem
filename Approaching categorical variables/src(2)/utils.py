import copy
from sklearn import preprocessing
import pandas as pd
import itertools

def feature_engineering(df,col):
    combination = list(itertools.combinations(col,2))
    for (c1,c2) in combination:
        df.loc[:,c1+'_'+c2] = df[c1].astype(str) + '_' + df[c2].astype(str)
    return df


def mean_encoding(data):
    # making a copy of df
    df = copy.deepcopy(data)

    num_cols = ['age','fnlwgt','capital.gain','capital.loss','hours.per.week']

    target_map = {'<=50K':0,
    '>50K':1}
    df.income = df.income.map(target_map)

    features = [f for f in df.columns if f not in ['kfold','income']]

    for col in features:
        if col not in num_cols:
            df.loc[:,col] = df[col].astype(str).fillna('NONE')
    
    for col in features:
        if col not in num_cols:
            lbl_encode = preprocessing.LabelEncoder()
            lbl_encode.fit(df[col])
            df.loc[:,col] = lbl_encode.transform(df[col])
    
    encoded_dfs = []

    for fold in range(5):
        df_train = df[df.kfold != fold]
        df_valid = df[df.kfold == fold]

        for feat in features:
            if feat not in num_cols:
                mapping_dict = dict(df_train.groupby(feat)['income'].mean())
                df_valid.loc[:,feat+'_enc'] = df_valid[feat].map(mapping_dict)
        
        encoded_dfs.append(df_valid)

    encoded_df = pd.concat(encoded_dfs,axis = 0)
    return encoded_df


