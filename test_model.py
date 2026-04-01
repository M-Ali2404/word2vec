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

if __name__ == "__main__":
    unittest.main()