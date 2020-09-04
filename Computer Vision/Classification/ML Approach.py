import os
import numpy as np
import pandas as pd
import joblib

from PIL import Image
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from tqdm import tqdm

def create_dataset(training_df, image_dir):
    """
    Takes in train dataframe and images directory and returns the 
    inputs and outputs for training

    :params training_df: dataframe with image ids and targets
    :params image_dir: The directory with images

    return X as inputs and y as targets
    """
    images = []
    targets = []

    for idx, row in tqdm(training_df.iterrows(), 
    total = len(training_df),
    desc='processing images'):
        image_path = os.path.join(image_dir,row['ImageId'] + '.png')
        image = Image.open(image_path)
        image = image.resize((256,256), resample = Image.BILINEAR)

        image = np.array(image).ravel()

        images.append(image)
        target = row['target']
        targets.append(target)

    images = np.array(images)
    targets = np.array(targets)

    print(images.shape)

    return images, targets


if __name__=='__main__':
    csv_path = '../Computer Vision/Data/train.csv'
    image_dir = '../Computer Vision/Data/train_png'

    df = pd.read_csv(csv_path)

    df['kfold'] = -1

    df = df.sample(frac = 1).reset_index(drop = True)
    y = df['target'].values
    kf = model_selection.StratifiedKFold(n_splits = 5)

    for f, (t_,v_) in enumerate(kf.split(df, y = y)):
        df.loc[v_,'kfold'] = f

    for fold in range(5):
        df_train = df[df['kfold'] != fold]
        df_test = df[df['kfold'] == fold]

        xtrain, ytrain = create_dataset(df_train, image_dir)
        xtest, ytest = create_dataset(df_test, image_dir)

        clf = ensemble.RandomForestClassifier(n_jobs = -1)
        clf.fit(xtrain, ytrain)

        preds = clf.predict_proba(xtest)[:,1]

        auc = metrics.roc_auc_score(ytest, preds)

        print(f'Fold: {fold}')
        print(f'AUC: {auc}')
        print()

        # saving model
        joblib.dump(clf, f'rf_{fold}_{auc}.bin')




    
