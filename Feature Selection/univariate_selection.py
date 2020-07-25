from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.datasets import fetch_california_housing

class UnivariateFeatureSelection:

    def __init__(self,n_features,scoring,problem_type):

        if problem_type == 'classification':
            valid_scoring = {'chi2':chi2,
            'f_classif':f_classif,
            'mutual_info_classif':mutual_info_classif}
        elif problem_type == 'regression':
            valid_scoring = {'f_regression':f_regression,
            'mutual_info_regression':mutual_info_regression}
        
        if scoring not in valid_scoring:
            raise Exception('Not a valid scoring option')
        
        if isinstance(n_features,int):
            self.select = SelectKBest(valid_scoring[scoring],
            k = n_features)
        elif isinstance(n_features,float):
            self.select = SelectPercentile(valid_scoring[scoring],
            percentile = int(n_features)*100)

        else:
            raise Exception('Invalid type of feature number')
    
    def fit(self,X,y):
        return self.select.fit(X,y)

    def transform(self,X,y = None):
        return self.select.transform(X)
    def fit_transform(self, X, y):
        return self.select.fit_transform(X,y)


data = fetch_california_housing()
X = data['data']
y = data['target']

UVFS = UnivariateFeatureSelection(n_features = 5,scoring = 'mutual_info_regression',
problem_type = 'regression')

UVFS.fit(X,y)
X_tr = UVFS.transform(X)

# better to create less and important features instead of a lot of features 
# univariate selection doen't work well often most of the times feature selection
# is done using the model performance itself