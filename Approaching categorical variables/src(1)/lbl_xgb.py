import pandas as pd
import config
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing
print('Working')
def run(fold):
    # reading the data
    df = pd.read_csv(config.train_fold_data)

    # selectinf the features
    features = [f for f in df.columns if f not in ['id','target','kfold']]

    # treating NANS
    for col in features:
        df.loc[:,col] = df[col].astype(str).fillna('NONE')

    # label encoding
    for feat in features:
        lbl_enc = preprocessing.LabelEncoder()
        lbl_enc.fit(df[feat])
        df.loc[:,feat] = lbl_enc.transform(df[feat])
    
    # splitting the data based on the folds created
    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    # xgb
    model = xgb.XGBClassifier(n_jobs = -1,
    max_depth = 7,
    n_estimators = 200)

    model.fit(x_train,df_train.target.values)

    # AUC

    # taking the probability of 1
    valid_pred = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(df_valid.target.values,
    valid_pred)

    print('Fold: ',fold,'Validation AUC: ',auc)

if __name__ == '__main__':
    for fold_ in range(5):
        run(fold_)


