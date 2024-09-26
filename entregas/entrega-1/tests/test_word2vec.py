import unittest
import numpy as np
from word2vec import *


class TestWord2Vec(unittest.TestCase):

    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertAlmostEqual(sigmoid(1), 0.7310585786300049)
        self.assertAlmostEqual(sigmoid(-1), 0.2689414213699951)

    def test_naiveSoftmaxLossAndGradient(self):
        dataset = type('dummy', (), {})()
        dataset.sampleTokenIdx = lambda: 0
        dataset.getRandomContext = lambda C: ('a', ['a', 'b', 'c', 'd'])

        centerWordVec = np.random.randn(3, )
        outsideVectors = np.random.randn(5, 3)
        outsideWordIdx = 0
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(centerWordVec, outsideWordIdx,
                                                                           outsideVectors, dataset)

        self.assertTrue(isinstance(loss, float))
        self.assertEqual(gradCenterVec.shape, centerWordVec.shape)
        self.assertEqual(gradOutsideVecs.shape, outsideVectors.shape)

    def test_skipgram(self):
        dataset = type('dummy', (), {})()
        dataset.sampleTokenIdx = lambda: 0
        dataset.getRandomContext = lambda C: ('a', ['a', 'b', 'c', 'd'])

        word2Ind = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
        centerWordVectors = np.random.randn(5, 3)
        outsideVectors = np.random.randn(5, 3)

        loss, gradCenterVecs, gradOutsideVectors = skipgram('a', 2, ['a', 'b', 'c', 'd'], word2Ind, centerWordVectors,
                                                            outsideVectors, dataset)

        self.assertTrue(isinstance(loss, float))
        self.assertEqual(gradCenterVecs.shape, centerWordVectors.shape)
        self.assertEqual(gradOutsideVectors.shape, outsideVectors.shape)


if __name__ == "__main__":
    unittest.main()
