import pandas as pd
import config
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics


def run(fold):
    df = pd.read_csv(config.train_data_folds)

    num_cols = ['age','fnlwgt','capital.gain','capital.loss','hours.per.week']

    df = df.drop(num_cols, axis = 1)

    target_map = {'<=50K':0,
    '>50K':1}
    df.income = df.income.map(target_map)

    features = [f for f in df.columns if f not in ['kfold','income']]

    for feat in features:
        df.loc[:,feat] = df[feat].astype(str).fillna('NONE')
    
    df_train = df[df.kfold!=fold]
    df_valid = df[df.kfold==fold]

    full_data = pd.concat([df_train[features],df_valid[features]],axis = 0)
    ohe = preprocessing.OneHotEncoder()

    ohe.fit(full_data)
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    model = linear_model.LogisticRegression()
    model.fit(x_train,df_train.income.values)
    valid_preds = model.predict_proba(x_valid)[:,1]

    auc = metrics.roc_auc_score(df_valid.income.values,valid_preds)

    print(f'Fold: {fold}, AUC: {auc}')


if __name__ == '__main__':
    for fold_ in range(5):
        run(fold_)



