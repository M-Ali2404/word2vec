import numpy as np
from model import sigmoid

def eval_model(model, vocab_map):
    print("\nsimilarity check ")

    test_pairs = [
            ("orange", "juice"),
            ("king", "book"),
            ("fox", "dog"),
            ("orange", "book"), 
            ("quick", "lazy")
        ]
        # loop through tests
    for word_a, word_b in test_pairs:
            # check if words are in our vocab
        if word_a in vocab_map and word_b in vocab_map:
            id_a = vocab_map[word_a]
            id_b = vocab_map[word_b]
            
            vec_a = model.input_matrix[id_a]
            vec_b = model.output_matrix[id_b]
            
            final_score = np.dot(vec_a, vec_b)
            similarity = sigmoid(final_score)
            print(f"similarity ({word_a} -> {word_b}): {similarity:.4f}")
