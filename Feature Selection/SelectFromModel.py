import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

data = load_diabetes()
X = data['data']
y = data['target']
columns = data['feature_names']

model = RandomForestRegressor(n_jobs = -1)

model.fit(X,y)

feature_importance = model.feature_importances_
idxs = np.argsort(feature_importance)

plt.barh(range(len(idxs)), feature_importance[idxs], align = 'center')
plt.yticks(range(len(idxs)), [columns[i] for i in idxs])
plt.xlabel('Feature Importance')
plt.title('Feature Importance Random Forest')
plt.show()

sfm = SelectFromModel(estimator = model)

X_transformed = sfm.fit_transform(X,y)
print(X.shape)
print(X_transformed.shape)


support  =sfm.get_support()
print([x for x,y in zip(columns, support) if y == True])


# here the default threshold is 1e-5 which 
# results in just 2 features getting selected


# in models with L1 regularisation, unimportant feature weights are pushed to
# 0, so they in feature selection non-zero features are taken.

# do feature selection on training and then validate on validation model to 
# prevent overfitting


