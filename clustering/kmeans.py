from numpy import zeros, sum, sqrt, argmin
from numpy.random import shuffle
from chart.kmeans_chart import KMeansChart


class KMeans:

    def __init__(self):
        self.chart = None
        pass

    def fit(self, x, k, max_iter, plot_progress=False):
        (m, n) = x.shape
        initial_centroids = self.init_centroids(x, k)
        centroids = initial_centroids
        previous_centroids = centroids
        idx = zeros(m)
        for i in range(max_iter):
            idx = self.findClosestCentroids(x, centroids)
            if plot_progress:
                self._plotProgressKMeans(x, centroids, previous_centroids, idx, k, i)
                previous_centroids = centroids
            centroids = self.computeCentroids(x, idx, k)
        if plot_progress:
            self._show()
        return idx, centroids

    def findClosestCentroids(self, x, centroids):
        k = centroids.shape[0]
        idx = zeros(x.shape[0])
        m = len(x)
        dis = zeros(k)
        for i in range(m):
            for j in range(k):
                dist = x[i, :] - centroids[j, :]
                dis[j] = sum(dist**2)
                dis[j] = sqrt(dis[j])
            idx[i] = argmin(dis)

        return idx

    def computeCentroids(self, x, idx, k):
        m, n = x.shape
        centroids = zeros((k, n))

        for i in range(k):
            c = sum(idx == i)
            centroids[i, :] = sum(x[idx == i, :], 0) / c

        return centroids

    def _plotProgressKMeans(self, x, centroids, previous_centroids, idx, k, i):
        if self.chart is None:
            self.chart = KMeansChart()

        self.chart.plotProgressKMeans(x, centroids, previous_centroids, idx, k, i)

    def _show(self):
        self.chart.show()

    def init_centroids(self, x, k):
        shuffle(x)
        return x[:k, :]
