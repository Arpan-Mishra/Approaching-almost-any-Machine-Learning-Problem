import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# loading dataset

data = fetch_california_housing()
X = data['data']
y = data['target']

# define model

model = LinearRegression()

# rfe
rfe = RFE(estimator = model,
n_features_to_select = 3)

rfe.fit(X,y)

X_transformed = rfe.transform(X)

print(X.shape)
print(X_transformed.shape)


