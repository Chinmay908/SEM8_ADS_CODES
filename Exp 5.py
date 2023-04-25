from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score

df = datasets.load_iris()
X = df.data
y = df.target

kmeans_model = KMeans(n_clusters=3,n_init="auto",random_state=1).fit(X)
y_kmeans = kmeans_model.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

#Plotting the centroids of the clusters
plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')

plt.legend()
# plt.show()

#Extrinsic Measures
ri = adjusted_rand_score(y,y_kmeans)
print("\nAdjusted Rand Index:",ri)

mi = normalized_mutual_info_score(y,y_kmeans)
print("\nMutual Information:",mi)

sil_score = silhouette_score(X,y_kmeans,metric="euclidean")
print("\nSilhouette Coeff:",sil_score)