import unittest
from utility.pca import PCA
from numpy import array, all, isclose

class PCATest(unittest.TestCase):

    def testFit(self):
        a = array([[1, 1], [2, 2]])
        u_expected = array([[-0.7071, -0.7071],
                            [-0.7071, 0.7071]])
        s_expected = array([5, 0])

        u, s = PCA().fit(a)
        self.assertTrue(all(isclose(u, u_expected, rtol=10**-4)))
        self.assertTrue(all(isclose(s, s_expected, )))

    def testProject(self):
        a = array([[1, 1], [2, 2]])
        u = array([[-0.7071, -0.7071],
                    [-0.7071, 0.7071]])
        actual = PCA().project(a, u, 1)
        expected = array([[-1.4142], [-2.8284]])
        self.assertTrue(all(actual == expected))

    def testRecover(self):
        expected = array([[1, 1], [2, 2]])
        u = array([[-0.7071, -0.7071],
                    [-0.7071, 0.7071]])
        z = array([[-1.4142], [-2.8284]])

        actual = PCA().recover(z, u, 1)

        self.assertTrue(all(isclose(actual, expected, rtol=10**-1)))
