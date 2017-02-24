import unittest
from numpy import array, all
from clustering.kmeans import KMeans

class KMeansTest(unittest.TestCase):

    def testFindClosestCentroids(self):
        x = array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [4, 5]])
        centroids = array([[0, 1],
                          [4, 5]])
        idx = array([0, 0, 0, 1])

        actual = KMeans().findClosestCentroids(x, centroids)
        self.assertTrue(all(actual == idx))

    def testComputeCentroids(self):
        x = array([[0, 1],
                   [1, 2],
                   [2, 3],
                   [4, 5]])
        centroids = array([[1., 2.],
                           [4., 5.]])
        idx = array([0, 0, 0, 1])

        actual = KMeans().computeCentroids(x, idx, 2)
        self.assertTrue(all(actual == centroids))