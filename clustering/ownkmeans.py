import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np


style.use('ggplot')


x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

colors = ['g', 'r', 'c', 'b', 'k']

class KMeans(object):
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = {}
        self.classifications = {}

    def fit(self, data):
        for i in range(self.k):
            self.centroids[i] = data[i]
        for i in range(self.max_iter):
            for j in range(self.k):
                self.classifications[j] = []
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification1 = distances.index((min(distances)))
                self.classifications[classification1].append(featureset)

            prev_centroids = dict(self.centroids)
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid * 100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index((min(distances)))
        return classification


clf = KMeans()
clf.fit(x)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], color='y', marker='o', s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], color=color, marker='x', linewidths=5, s=150)

predict_data = np.array([[11, 4], [5, 2], [1, 10], [7, 7], [1, 0]])

for predict in predict_data:
    classification = clf.predict(predict)
    plt.scatter(predict[0], predict[1], s=125, marker='*', color=colors[classification])

plt.show()
