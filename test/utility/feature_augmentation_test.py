import unittest
from numpy import array, all
from utility import FeatureAugmentation as fa

class FeatureAugmentationTest(unittest.TestCase):

    def setUp(self):
        self.x = array([[2, 3, 5],
               [2, 3, 5]])
        self.expected = array([[6, 10, 15, 2, 3, 5, 4, 9, 25],
                               [6, 10, 15, 2, 3, 5, 4, 9, 25]])
        self.expected2 = array([[2, 3, 5, 4, 9, 25],
                                [2, 3, 5, 4, 9, 25]])
        self.expected3 = array([[2, 3, 5, 4, 9, 25, 8, 27, 125],
                                [2, 3, 5, 4, 9, 25, 8, 27, 125]])
        self.expected4 = array([[6, 12, 18, 10, 20, 50, 15, 45, 75,
                                 2, 3, 5, 4, 9, 25, 8, 27, 125],
                                [6, 12, 18, 10, 20, 50, 15, 45, 75,
                                 2, 3, 5, 4, 9, 25, 8, 27, 125]])

    def tearDown(self):
        self.x = None
        self.expected2 = None

    def testMap(self):
        actual = fa.map(self.x)
        self.assertEqual(actual.shape[1], self.expected.shape[1])
        self.assertTrue(all(actual == self.expected))
        actual = fa.map(self.x, degree=3)
        self.assertEqual(actual.shape[1], self.expected4.shape[1])
        self.assertTrue(all(actual == self.expected4))

    def testMap2(self):
        actual2 = fa.map2(self.x)
        self.assertEqual(actual2.shape[1], self.expected2.shape[1])
        self.assertTrue(all(actual2 == self.expected2))
        actual = fa.map2(self.x, degree=3)
        self.assertEqual(actual.shape[1], self.expected3.shape[1])
        self.assertTrue(all(actual == self.expected3))

    def testMapDegree1(self):
        actual = fa.map(self.x, degree=1)
        self.assertEqual(actual.shape[1], self.x.shape[1])
        self.assertTrue(all(actual == self.x))

    def testMap2Degree1(self):
        actual2 = fa.map2(self.x, degree=1)
        self.assertEqual(actual2.shape[1], self.x.shape[1])
        self.assertTrue(all(actual2 == self.x))
