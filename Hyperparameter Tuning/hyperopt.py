import numpy as np
import pandas as pd

import functools

from sklearn import ensemble, metrics, model_selection

from hyperopt import hp, fmin, tpe, Trials
from hyperopt import pyll

def optimize(params, x, y):
    """
    :params params: list of params from gp_minimize
    :params x: training data
    :params y: labels/targets

    :returns: negative accuracy after 5-folds
    """

   
    model = ensemble.RandomForestClassifier(**params) 
    kf = model_selection.StratifiedKFold(n_splits=5) 
    accuracies = []    
    for idx in kf.split(X=x, y=y):  
        train_idx, test_idx = idx[0], idx[1]  
        xtrain = x[train_idx]  
        ytrain = y[train_idx]  
        xtest = x[test_idx]  
        ytest = y[test_idx]  
        model.fit(xtrain, ytrain) 
        preds = model.predict(xtest)    
        fold_accuracy = metrics.accuracy_score(  ytest,  preds  )  
        accuracies.append(fold_accuracy) 
        return -1 * np.mean(accuracies) 


if __name__ == '__main__':
    
    df = pd.read_csv('Hyperparameter Tuning/mobile_train.csv')

    X = df.drop('price_range', axis = 1).values
    y = df.price_range.values

    param_space = {
        'max_depth': pyll.scope.int(hp.quniform('max_depth', 1, 15, 1)),
        'n_estimators': pyll.scope.int(hp.quniform('n_estimators', 10, 1500, 1)),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_features': hp.uniform('max_features', 0, 1)
    }

    optmization_function = functools.partial(optimize,
    x = X,
    y = y)

    trials = Trials()

    hopt = fmin(fn = optmization_function,
    space = param_space,
    algo = tpe.suggest,
    max_evals=15,
    trials=trials)

    print(hopt)

