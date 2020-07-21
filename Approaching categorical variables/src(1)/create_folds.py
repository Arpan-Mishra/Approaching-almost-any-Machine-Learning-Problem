import pandas as pd
import config
from sklearn import model_selection

if __name__ == '__main__':
    df = pd.read_csv(config.train_data)
    df['kfold'] = -1

    df = df.sample(frac = 1).reset_index(drop = True)

    y = df.target.values

    kf = model_selection.StratifiedKFold(n_splits = 5)

    for f,(t_,v_) in enumerate(kf.split(X = df,y = y)):
        df.loc[v_,'kfold'] = f
    
    df.to_csv(config.train_fold_data,index = False)

    # target distribution per fold is going to be same and 
    # the total data points will also be similar