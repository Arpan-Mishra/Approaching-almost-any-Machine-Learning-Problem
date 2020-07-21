import pandas as pd
import config
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing
def run(fold):
    # reading the data
    df = pd.read_csv(config.train_data_folds)
    
    num_cols = ['age','fnlwgt','capital.gain','capital.loss','hours.per.week']
    
    target_map = {'<=50K':0,
    '>50K':1}
    df.income = df.income.map(target_map)

    # selecting the features
    features = [f for f in df.columns if f not in ['income','kfold']]

    # treating NANS
    for col in features:
        if col not in num_cols:
            df.loc[:,col] = df[col].astype(str).fillna('NONE')

    # label encoding
    for feat in features:
        if col not in num_cols:
            lbl_enc = preprocessing.LabelEncoder()
            lbl_enc.fit(df[feat])
            df.loc[:,feat] = lbl_enc.transform(df[feat])
    
    # splitting the data based on the folds created
    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    # xgb
    model = xgb.XGBClassifier(n_jobs = -1)

    model.fit(x_train,df_train.income.values)

    # AUC

    # taking the probability of 1
    valid_pred = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(df_valid.income.values,
    valid_pred)

    print('Fold: ',fold,'Validation AUC: ',auc)

if __name__ == '__main__':
    for fold_ in range(5):
        run(fold_)


