import unittest
from test.utility.feature_augmentation_test import FeatureAugmentationTest
from test.utility.pca_test import PCATest

test_classes_to_run = [FeatureAugmentationTest, PCATest]
def create_suite():
    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    return unittest.TestSuite(suites_list)

if __name__ == '__main__':
    suite = create_suite()
    runner = unittest.TextTestRunner()
    runner.run(suite)
