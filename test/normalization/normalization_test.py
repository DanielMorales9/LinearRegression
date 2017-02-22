import unittest
from numpy import array, all, isclose

from normalization.normalization import MeanNormalization
from normalization.normalization import ZScoreNormalization


class NormalizationTest(unittest.TestCase):

    def testMeanNormalizationMatrix(self):
        x = array([[1, 2, 3],
                    [0, 1, 2]])
        actual = MeanNormalization().scale(x)

        expected = array([[0.5, 0.5, 0.5],
                          [-0.5, -0.5, -0.5]])
        self.assertTrue(all(actual == expected))

    def testMeanNormalizationVector(self):
        x = array([1, 2, 3])
        actual = MeanNormalization().scale(x)
        expected = array([-0.5, 0, 0.5])
        self.assertTrue(all(actual == expected))

    def testMeanNormalizationVectorNoRangeColumn(self):
        x = array([1, 1, 1])
        actual = MeanNormalization().scale(x)

        expected = array([0, 0, 0])
        self.assertTrue(all(actual == expected))

    def testMeanNormalizationMatrixNoRangeColumn(self):
        x = array([[1, 2, 3],
                   [1, 1, 2]])
        actual = MeanNormalization().scale(x)

        expected = array([[0, 0.5, 0.5],
                          [0, -0.5, -0.5]])
        self.assertTrue(all(actual == expected))

    def testZScoreMatrix(self):
        x = array([[1, 2, 3],
                    [0, 1, 2]])
        actual = ZScoreNormalization().scale(x)

        expected = array([[1., 1., 1.],
                          [-1., -1., -1.]])
        self.assertTrue(all(actual == expected))

    def testZScoreVector(self):
        x = array([1, 2, 3])
        actual = ZScoreNormalization().scale(x)

        expected = array([-1.22474487,  0.,  1.22474487])
        self.assertTrue(all(isclose(actual, expected)))

    def testZScoreVectorNoRangeColumn(self):
        x = array([1, 1, 1])
        actual = MeanNormalization().scale(x)

        expected = array([0, 0, 0])
        self.assertTrue(all(actual == expected))

    def testZScoreMatrixNoRangeColumn(self):
        x = array([[1, 2, 3],
                   [1, 1, 2]])
        actual = MeanNormalization().scale(x)
        expected = array([[0, 0.5, 0.5],
                          [0, -0.5, -0.5]])
        self.assertTrue(all(actual == expected))
