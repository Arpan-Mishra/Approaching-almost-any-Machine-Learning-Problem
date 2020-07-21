import pandas as pd
import config
import utils
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing


def run(df,fold):
    
    # selecting the features
    features = [f for f in df.columns if f not in ['income','kfold']]

        
    # splitting the data based on the folds created
    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    # xgb
    model = xgb.XGBClassifier(n_jobs = -1,max_depth = 7,n_estimators = 200)

    model.fit(x_train,df_train.income.values)

    # AUC

    # taking the probability of 1
    valid_pred = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(df_valid.income.values,
    valid_pred)

    print('Fold: ',fold,'Validation AUC: ',auc)

if __name__ == '__main__':
    df = pd.read_csv(config.train_data_folds)
    df = utils.mean_encoding(df)
    for fold_ in range(5):
        run(df,fold_)

# target encoding prone to overfitting
# try using smoothing in sklearn contrib package, this introduces regularisation effect
