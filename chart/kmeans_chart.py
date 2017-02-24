import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from numpy import arange, ones
import numpy as np

class KMeansChart:

    def plotProgressKMeans(self, x, centroids, previous_centroids, idx, k, i):

        self.plotDataPoints(x, idx, k)

        plt.plot(centroids[:, 0], centroids[:, 1], 'x', markeredgecolor='k', markersize=10, linewidth=3)
        for j in range(centroids.shape[0]):
            x1 = centroids[j, 0]
            x2 = previous_centroids[j, 0]
            y1 = centroids[j, 1]
            y2 = previous_centroids[j, 1]

            plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
        plt.title("Iteration {}".format(i+1))

    def show(self):
        plt.show()

    def plotDataPoints(self, x, idx, k):
        idx = idx.astype(int)
        palette = self._hsv(float(k))
        colors = palette[idx, :]
        plt.scatter(x[:, 0], x[:, 1],  s=15, c=colors)

    def _hsv(self, k):
        h = arange(0, k) / k
        H = ones((int(k), 3))
        H[:, 0] = h
        return hsv_to_rgb(H)
