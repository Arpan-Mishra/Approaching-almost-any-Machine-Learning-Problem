import pandas as pd
import config
from scipy import sparse
from sklearn import ensemble
from sklearn import decomposition
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

    # splitting the data based on the folds created
    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)

    # OHE
    ohe = preprocessing.OneHotEncoder()

    # concatenating data
    full_data = pd.concat([df_train[features],df_valid[features]],axis = 0)
    
    ohe.fit(full_data[features])

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    # SVD - 120 components
    svd = decomposition.TruncatedSVD(n_components = 120)
    full_sparse = sparse.vstack((x_train,x_valid))
    svd.fit(full_sparse)

    x_train = svd.transform(x_train)
    x_valid = svd.transform(x_valid)

    # Random Forest
    model = ensemble.RandomForestClassifier(n_jobs = -1)
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


