import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets
from sklearn import manifold
from sklearn.cluster import KMeans
%matplotlib inline

# fetching data
data = datasets.fetch_openml('mnist_784',
                             version = 1,
                             return_X_y = True)

pixel_values, targets = data
targets = targets.astype(int)

# visualising image
single_image = pixel_values[1,:].reshape(28,28)

plt.imshow(single_image,cmap = 'gray')

# tsne
tsne = manifold.TSNE(n_components = 2, random_state = 42)
transformed_data = tsne.fit_transform(pixel_values[:3000,:]) # taking only 3k points

tsne_df = pd.DataFrame(np.column_stack((transformed_data,targets[:3000])),
                       columns = ['x','y','targets'])

tsne_df.loc[:,targets] = tsne_df.targets.astype(int)


grid = sns.FacetGrid(tsne_df, hue = 'targets',size = 8)
grid.map(plt.scatter,'x','y').add_legend()

# K-means

kmeans = KMeans(n_clusters=10,n_jobs = -1)
kmeans.fit(transformed_data)
clusters = kmeans.predict(transformed_data)

plt.scatter(*transformed_data.T,c = clusters, cmap='viridis')
plt.show()
