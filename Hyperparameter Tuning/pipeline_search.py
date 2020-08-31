# pipeline_search.py

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline, preprocessing

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

def quadratic_wighted_kappa(y_true, y_pred):
    
    """
    Wrapper for cohen's kappa with quadratic weights
    """

    return metrics.cohen_kappa_score(y_true, y_pred, weights = 'quadratic')

if __name__ == '__main__':
    
    # load train file
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    idx = train.id.values.astype(int)
    train = train.drop('id', axis = 1)
    test = test.drop('id', axis = 1)

    y = train.relevance.values

    traindata = list(train.apply(lambda x: '%s %s ' % (x['text1'], x['text2'])), axis = 1)
    testdata = list(test.apply(lambda x: '%s %s ' % (x['text1'], x['text2'])), axis = 1)

tfv = TfidfVectorizer(
    min_df = 3,
    max_features = None,
    strip_accents = 'unicode',
    token_pattern = r'\w{1,}',
    ngram_range = (1,3),
    use_idf = 1,
    smooth_idf = 1,
    sublinear_tf = 1,
    stop_words = 'english'
)

tfv.fit(traindata)
X = tfv.transform(traindata)

X_test = tfv.transform(testdata)


svd = TruncatedSVD()

scl = preprocessing.StandardScaler()

svm_model = SVC()

clf = pipeline.Pipeline([
    ('svd', svd),
    ('scl', scl),
    ('svm', svm_model)
])

param_grid = {'svd__n_components':[200,300],
'svm__C':[10,12]}

kappa_scorer = metrics.make_scorer(quadratic_wighted_kappa,
greater_is_better = True)

model = model_selection.GridSearchCV(
    estimator = clf,
    param_grid=param_grid,
    scoring=kappa_scorer,
    verbose = 10,
    n_jobs = -1,
    refit = True,
    cv = 5
)

model.fit(X,y)
print('Best score: %0.3f' % model.best_score_)
print('Best parameters set:')
best_parameters = model.best_estimator_.get_params()

for param_name in sorted(param_grid.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))

best_model = model.best_estimator_

best_model.fit(X,y)
preds = best_model.predict(X_test)










