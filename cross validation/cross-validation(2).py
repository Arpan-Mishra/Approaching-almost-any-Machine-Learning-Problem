# K-fold cross validaiton - general framework

import pandas as pd
import seaborn as sns
from sklearn import model_selection

if __name__ == '__main__':
     df = pd.read_csv('train.csv')
    
     df['kfold'] = -1
     
     df = df.sample(frac = 1).reset_index(drop = True)
     
     kf = model_selection.KFold(n_splits = 5)
     
     for fold,(trn_,val_) in enumerate(kf.split(X = df)):
         df.loc[val_,'kfold'] = fold
         
     # save the new csv file with kfold cols
     
     df.to_csv('train_kfols.csv',index = False)



# stratified k-fold

if __name__ == '__main__':
     df = pd.read_csv('train.csv')
    
     df['kfold'] = -1
     
     df = df.sample(frac = 1).reset_index(drop = True)
     
     y = df.target.values
     
     kf = model_selection.StratifiedKFold(n_splits = 5)
     
     for fold,(trn_,val_) in enumerate(kf.split(X = df,y = y)):
         df.loc[val_,'kfold'] = fold
         
     # save the new csv file with kfold cols
     
     df.to_csv('train_kfols.csv',index = False)

     
# distribution of labels of wine dataset      
     
# distribution of labels

b = sns.countplot(x = 'quality',data = df)
b.set_xlabel('quality',fontsize = 20)
b.set_ylabel('count',fontsize = 20)     
     
# usage tips
# 1. use stratified for standard classification
# 2. if data very large use hold out
# 3. if data very small leave one sample for validation rest for training in all folds (k = N)
# 4. time series - hold out

# regression
#1. all but stratified can be used as it is
# 2. stratified can be used by changing the problem a bit
# 3. can be used when distribution of targets is not consistent
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     