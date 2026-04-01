import unittest
import numpy as np
from model import SkipGramModel, sigmoid

class TestWord2VecEngine(unittest.TestCase):
    
    def test_model_initialization(self):
        vocab_size = 10
        embed_dim = 5
        learning_rate = 0.01
        
        model = SkipGramModel(vocab_size, embed_dim, learning_rate)
        self.assertEqual(model.input_matrix.shape, (10, 5))
        self.assertEqual(model.output_matrix.shape, (10, 5))

    def test_sigmoid_math(self):
        result = sigmoid(0)
        self.assertEqual(result, 0.5)

    def test_loss_calculation(self):
        from model import bce_loss
        loss_good = bce_loss(0.9, 1)
        loss_bad = bce_loss(0.1, 1)
        self.assertTrue(loss_bad > loss_good)
        self.assertTrue(loss_good > 0)

    def test_bce_math(self):
        from model import bce_loss

        loss_near_perfect = bce_loss(1, 0.999999)
        self.assertAlmostEqual(loss_near_perfect, 0, places=4)

        loss_terrible = bce_loss(1, 0.0001)
        self.assertTrue(loss_terrible > 5.0)
        
        loss_half = bce_loss(1, 0.5)
        self.assertAlmostEqual(loss_half, 0.693147, places=5)

if __name__ == "__main__":
    unittest.main()