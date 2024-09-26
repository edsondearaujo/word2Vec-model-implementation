import unittest
from sgd import sgd


class TestSGD(unittest.TestCase):

    def test_sgd(self):
        gradients = [1.0, 2.0, 3.0]
        learning_rate = 0.1
        updated_params = sgd(gradients, learning_rate)

        expected_params = [-0.1, -0.2, -0.3]
        for updated, expected in zip(updated_params, expected_params):
            self.assertAlmostEqual(updated, expected, places=7)


if __name__ == "__main__":
    unittest.main()
