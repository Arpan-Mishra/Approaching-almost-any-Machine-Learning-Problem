import pandas as pd
from sklearn import model_selection
import config

if __name__ == '__main__':
     df = pd.read_csv(config.train_input)
    
     df['kfold'] = -1
     
     df = df.sample(frac = 1).reset_index(drop = True)
     
     y = df.income.values
     
     kf = model_selection.StratifiedKFold(n_splits = 5)
     
     for fold,(trn_,val_) in enumerate(kf.split(X = df,y = y)):
         df.loc[val_,'kfold'] = fold
         
     # save the new csv file with kfold cols
     
     df.to_csv(config.train_data_folds,index = False)
     
     