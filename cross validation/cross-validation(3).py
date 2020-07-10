# stratified k-fold for regression

# first divide into bins then apply stratified k-fold
# use sturges rule for deciding num bins if sample is not large

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection

def create_folds(data):
    data['kfold'] = -1
    
    # shuffle
    data = data.sample(frac = 1).reset_index(drop = True)
    
    # sturge rule
    num_bins = int(np.floor(1 + np.log2(len(data))))
    # bin targets
    data.loc[:,'bins'] = pd.cut(data['target'],bins = num_bins, labels = False)
    
    # k fold 
    kf = model_selection.StratifiedKFold(n_splits = 5)
    
    # use bins for stratification
    for f, (t,v) in enumerate(kf.split(X = data, y = data.bins.values)):
        data.loc[v,'kfold'] = f
        
    # drop bins
    data = data.drop('bins',axis = 1)
    
    return data


if __name__ == '__main__':
    # creating a sample dataset
    X,y = datasets.make_regression(n_samples = 15000, 
                                   n_features = 100, n_targets = 1)
    
    df = pd.DataFrame(X, columns = [f"f_{i}" for i in range(X.shape[1])])
    df.loc[:,'target'] = y
    # createing folds
    df = create_folds(df)
    
    
# if there are groups in the dataset, for example patients we can try GroupKfold    
    
    
    
    
    
    
    
    
    
    
    