from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
               [4, 2], [4, 4], [4, 0]])
data = np.loadtxt('Train1.csv', delimiter=',')
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
print kmeans.cluster_centers_
