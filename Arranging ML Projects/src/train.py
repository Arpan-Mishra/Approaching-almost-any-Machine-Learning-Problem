import joblib
import config
import argparse
import pandas as pd
import os
import model_dispatcher
from sklearn import metrics


def run(fold,model):
    # read data
    df = pd.read_csv(config.input_train)
    
    # splitting data
    df_train = df[df.kfold!=fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)
    
    x_train = df_train.drop('label',axis = 1).values
    y_train = df_train.label.values

    x_valid = df_valid.drop('label',axis = 1).values  
    y_valid = df_valid.label.values
    
    # fitting and prediction
    clf = model_dispatcher.models[model]
    
    clf.fit(x_train,y_train)
    
    preds = clf.predict(x_valid)
    
    # evaluation
    accuracy = metrics.accuracy_score(y_valid,preds)
    print(f'fold = {fold}, Accuracy = {accuracy}')
    
    # save model
    joblib.dump(clf,os.path.join(config.model_path,
                                 f'dt_{fold}.bin'))
    

if __name__ == '__main__':
    # init Argument Parser
    parser = argparse.ArgumentParser()
    
    # adding arguments and type
    parser.add_argument(
            '--fold',
            type = int            
            )
    parser.add_argument(
            '--model',
            type = str
            )
    
    # read the fold specified in the command line
    args = parser.parse_args()
    # run the fold
    run(fold = args.fold, model = args.model)

    
    