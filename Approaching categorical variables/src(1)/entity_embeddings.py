import os
import gc
import joblib
import pandas as pd
import config
import numpy as np
from sklearn import metrics,preprocessing
from tensorflow.keras import layers,optimizers
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def create_model(df,cat_cols):

    inputs = []
    outputs = []

    for c in cat_cols:
        num_unique_vals = int(df[c].nunique())

        embed_dim = int(min(np.ceil((num_unique_vals)/2), 50)) 

        inp = layers.Input((1,))

        out = layers.Embedding(  num_unique_vals + 1, embed_dim, name=c  )(inp) 

        out = layers.SpatialDropout1D(0.3)(out)
        out = layers.Reshape(target_shape = (embed_dim,))(out)

        inputs.append(inp)
        outputs.append(out)   

    x = layers.Concatenate()(outputs)
    x = layers.BatchNormalization()(x)


    x = layers.Dense(300, activation = 'relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation = 'relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    y = layers.Dense(2,activation = 'softmax')(x)

    model = Model(inputs = inputs, outputs = y)

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    return model

def run(fold):
    df = pd.read_csv(config.train_fold_data)
    cat_cols = [f for f in df.columns if f not in ['id','target','kfold']]

   
    for c in cat_cols:
        df.loc[:,c] = df[c].astype(str).fillna('NONE')

    for c in cat_cols:
        lbl_enc = preprocessing.LabelEncoder()
        lbl_enc.fit(df[c])
        df.loc[:,c] = lbl_enc.transform(df[c].values)
    
    df_train = df[df.kfold!=fold].reset_index(drop = True)
    df_valid = df[df.kfold==fold].reset_index(drop = True)

    model = create_model(df,cat_cols)

    xtrain = [df_train[cat_cols].values[:, k] for k in range(len(cat_cols))]  
    xvalid = [df_valid[cat_cols].values[:, k] for k in range(len(cat_cols))] 



    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)

    model.fit(xtrain,ytrain_cat,validation_data = (xvalid,yvalid_cat),    
    batch_size = 1024,
    epochs = 3)

    valid_preds = model.predict(xvalid)[:,1]

    print(metrics.roc_auc_score(yvalid,valid_preds))

    K.clear_session()


if __name__ == '__main__':
    run(0)
    run(1)
    run(2)
    run(3)
    run(4)


# works very well when there are a lot of categories and we have sufficient data
# if we use OHE on 