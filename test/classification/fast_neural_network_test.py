import unittest
from numpy import arange, cos, array, all, isclose, ones
from classification import FastNeuralNetwork as nn

class FastNeuralNetworkTest(unittest.TestCase):

    def setUp(self):
        self.theta = arange(1, 19) / 10.0
        self.il = 2
        self.hl = 2
        self.nl = 4
        X = cos([[1, 2], [3, 4], [5, 6]])
        self.X = nn.reshape_training_set(X)
        y = (array([4, 2, 3]) - 1) % 4
        self.y = nn.reshape_labels(y)
        self.th1 = array([[0.1, 0.3, 0.5],
                              [0.2, 0.4, 0.6]])
        self.th2 = array([[0.7, 1.1, 1.5],
                              [0.8, 1.2, 1.6],
                              [0.9, 1.3, 1.7],
                              [1., 1.4, 1.8]])
        self.z2 = array([[0.05401727, 0.16643282],
                         [-0.52381956, -0.58818317],
                         [0.6651838, 0.88956705]])
        self.a2 = array([[0.51350103, 0.54151242],
                         [0.37195952, 0.35705182],
                         [0.66042389, 0.70880081]])
        self.a3 = array([[0.8886593, 0.9074274, 0.9233049, 0.9366493],
                         [0.8381779, 0.8602820, 0.8797997, 0.8969177],
                         [0.9234142, 0.9385775, 0.9508982, 0.9608506]])
        self.nn = nn()

    def tearDown(self):
        self.theta = None
        self.il = None
        self.hl = None
        self.nl = None
        self.X = None
        self.y = None
        self.a2 = None
        self.a3 = None
        self.z2 = None
        self.th1 = None
        self.th2 = None

    def testForwardPropagation(self):
        th1, th2, z2, aa2, a3 \
            = self.nn.forward_propagation(self.theta, self.il, self.hl,
                                       self.nl, self.X)
        self.assertTrue(all(isclose(z2, self.z2)))
        self.assertTrue(all(isclose(th2, self.th2)))
        self.assertTrue(all(isclose(th1, self.th1)))
        self.assertTrue(all(isclose(aa2[:, 1:], self.a2)))
        self.assertTrue(all(isclose(a3, self.a3)))

    def testCost(self):
        j = self.nn.cost(self.a3, self.y, self.th1, self.th2, 0)
        self.assertTrue(all(isclose([j], [7.4069])))

    def testCostRegularization(self):
        j = self.nn.cost(self.a3, self.y, self.th1, self.th2, 4.0)
        self.assertTrue(all(isclose([j], [19.473636522732416])))

    def testGradient(self):
        aa2 = ones((self.a2.shape[0], self.a2.shape[1]+1))
        aa2[:, 1:] = self.a2
        expected = array([0.766138369630136, 0.979896866040661,
                          -0.027539615885635, -0.035844208951086,
                          -0.024928782740987, -0.053861693972528,
                          0.883417207679397, 0.568762344914511,
                          0.584667662135129, 0.598139236978449,
                          0.459313549296372, 0.344618182524694,
                          0.256313331202455, 0.311885062878785,
                          0.478336623152867, 0.368920406686281,
                          0.259770621934099, 0.322330889109923])
        actual = self.nn.gradient(self.X, self.y, 0, self.th1, self.th2, self.a3, aa2, self.z2)
        self.assertTrue(all(isclose(actual, expected)))

    def testGradientRegularization(self):

        aa2 = ones((self.a2.shape[0], self.a2.shape[1]+1))
        aa2[:, 1:] = self.a2
        expected = array([0.766138369630136, 0.979896866040661,
                          0.372460384114365, 0.497489124382247,
                          0.641737883925680, 0.746138306027472,
                          0.883417207679397, 0.568762344914511,
                          0.584667662135129, 0.598139236978449,
                          1.925980215963038, 1.944618182524693,
                          1.989646664535788, 2.178551729545452,
                          2.478336623152867, 2.502253740019614,
                          2.526437288600766, 2.722330889109923])
        actual = self.nn.gradient(self.X, self.y, 4.0, self.th1, self.th2, self.a3, aa2, self.z2)
        self.assertTrue(all(isclose(actual, expected)))

    def testCostFunction(self):
        self.nn.l = 0
        self.nn.il = self.il
        self.nn.hl = self.hl
        self.nn.nl = self.nl
        self.nn.X = self.X
        self.nn.y = self.y
        expected = array([0.766138369630136, 0.979896866040661,
                          -0.027539615885635, -0.035844208951086,
                          -0.024928782740987, -0.053861693972528,
                          0.883417207679397, 0.568762344914511,
                          0.584667662135129, 0.598139236978449,
                          0.459313549296372, 0.344618182524694,
                          0.256313331202455, 0.311885062878785,
                          0.478336623152867, 0.368920406686281,
                          0.259770621934099, 0.322330889109923])
        j, grad = self.nn.cost_function(self.theta)
        self.assertTrue(all(isclose([j], [7.4069])))
        self.assertTrue(all(isclose(grad, expected)))

    def testCostFunctionWRegularization(self):
        self.nn.l = 4.0
        self.nn.il = self.il
        self.nn.hl = self.hl
        self.nn.nl = self.nl
        self.nn.X = self.X
        self.nn.y = self.y
        expected = array([0.766138369630136, 0.979896866040661,
                          0.372460384114365, 0.497489124382247,
                          0.641737883925680, 0.746138306027472,
                          0.883417207679397, 0.568762344914511,
                          0.584667662135129, 0.598139236978449,
                          1.925980215963038, 1.944618182524693,
                          1.989646664535788, 2.178551729545452,
                          2.478336623152867, 2.502253740019614,
                          2.526437288600766, 2.722330889109923])
        j, grad = self.nn.cost_function(self.theta)
        self.assertTrue(all(isclose([j], [19.473636522732416])))
        self.assertTrue(all(isclose(grad, expected)))
