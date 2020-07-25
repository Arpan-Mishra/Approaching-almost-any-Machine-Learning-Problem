# remove features with very low variance (this means that they are almost constant,
# and they do not add much value)

from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

data = ...
var_thresh = VarianceThreshold(threshold = 0.1)
transformed_data = var_thresh.fit_transform(data)

# another way is to remove highly correlated columns

data = fetch_california_housing()
X = data['data']
col_names = data['feature_names']
y = data['target']

df = pd.DataFrame(X, columns = col_names)
# introducing a highly correlated columns
df['medinc_sqr'] = df.MedInc.apply(np.sqrt)

df.corr()

# medinc_sqr is highly correlated with medinc so we can remove one of them

## Univatiate Feature Selection
# 1. Mutual Information
# 2. ANOVA F-test
# 3. chi square

# in sklearn  - SelectKBest or SelectPercentile



