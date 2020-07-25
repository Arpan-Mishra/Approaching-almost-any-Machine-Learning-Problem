import pandas as pd
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn import linear_model
import matplotlib.pyplot as plt 

class GreedyFeatureSelection:
    def evaluate_score(self, X,y):
        model = linear_model.LogisticRegression()
        model.fit(X,y)
        preds = model.predict_proba(X)[:,1]
        
        return metrics.roc_auc_score(y,preds)

    def _feature_selection(self, X,y):
        good_features = []
        best_scores = []
        
        while True:
            num_feature = X.shape[1]
            for feature in range(num_feature):
                best_score = 0
                curr_feature = None

                if feature in good_features:
                    continue

                features = good_features + [feature]
                xtrain = X[:,features]
                score = self.evaluate_score(xtrain,y)

                if score > best_score:
                    best_score = score
                    curr_feature = feature
            
            if curr_feature!=None:
                good_features.append(curr_feature)
                best_scores.append(best_score)

            if len(best_scores)>2:
                if best_score[-1]<best_score[-2]:
                    break
        return best_scores[:-1],good_features[:-1]

    def __call__(self,X,y):
        scores,features = self._feature_selection(X,y)

        return X[:,features], scores

X,y = make_classification(n_samples = 1000,n_features = 100)
X_transformed, scores = GreedyFeatureSelection()(X,y)

print(X.shape)
print(X_transformed.shape)
print(scores)

# the greedy approach is very time taking obviously so if we have a lot of features 
# and a lot of samples then it is better to not go with it
# also the AUC we are getting is overfitted so this is not the actual AUC but 
# being used only to determine features.


    
