import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans


style.use('ggplot')


x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

clf = KMeans(n_clusters=2)
clf.fit(x)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ['g.', 'r.', 'c.', 'b.', 'k.']

for i in range(len(x)):
    plt.plot(x[i][0], x[i][1], colors[labels[i]], markersize=25)
plt.scatter(centroids[:, 0], centroids[:, 1], color='y', marker='x', s=150, linewidths=5)

features = np.array([[10, 2], [0, 5]])
predict = clf.predict(features)
for i in range(len(features)):
    plt.plot(features[i][0], features[i][1], colors[predict[i]], marker='*', markersize=10)
plt.show()
