import numpy as np
from dataset import prepare_text_data
def sigmoid(raw_logit) -> float:
    # clip the value to avoid overflow error
    # if the number is too big exp explodes
    safe_z = np.clip(raw_logit, -32, 32)
    return 1.0 / (1.0 + np.exp(-safe_z))

def forwardpress(center_vector, context_vector):
    #doing the dot product here
    return np.dot(center_vector, context_vector)

def bce_loss(target_label, prediction_prob):
    # add a tiny epsilon to stop log(0) errors
    epsilon_val = 1e-9
    term_one = target_label * np.log(prediction_prob + epsilon_val)
    term_two = (1 - target_label) * np.log(1 - prediction_prob + epsilon_val)
    return -(term_one + term_two)

def backpress(center_vec, context_vec, prediction, target_label, learning_rate):
    error_diff = prediction - target_label 
    
    # calculate the gradients
    grad_center = error_diff * context_vec
    grad_context = error_diff * center_vec
    
    # update the vectors
    center_vec -= learning_rate * grad_center
    context_vec -= learning_rate * grad_context
    
    return center_vec, context_vec

class SkipGramModel:
    def __init__(self, vocab_size, embed_dim, learning_rate):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        
        np.random.seed(42)
        # initialise weights randomly
        self.input_matrix = np.random.uniform(-0.5, 0.5, (vocab_size, embed_dim))
        self.output_matrix = np.random.uniform(-0.5, 0.5, (vocab_size, embed_dim))