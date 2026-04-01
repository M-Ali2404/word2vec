import numpy as np

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
# setup the text data
raw_text = "orange juice king book the of the quick brown fox jumps over the lazy dog"
# split the string manually
tokens = raw_text.lower().split()

# get unique words for vocab
unique_words = sorted(list(set(tokens)))

# build the maps
vocab_map = {}
reverse_vocab_map = {}

for index, word in enumerate(unique_words):
    vocab_map[word] = index
    reverse_vocab_map[index] = word

# convert text to a list of ids
word_id_list = []
for t in tokens:
    word_id_list.append(vocab_map[t])

print(f"Vocab Size: {len(unique_words)}")
print(f"Training on: {tokens}")

##hyperparameters
VOCAB_SIZE = len(unique_words)
EMBED_DIM = 10     # dimension of the vector
LEARNING_RATE = 0.01
EPOCHS = 1000      
WINDOW_SIZE = 2    # look 2 words left and right

# initialise weights randomly
np.random.seed(42) 
input_matrix = np.random.uniform(-0.5, 0.5, (VOCAB_SIZE, EMBED_DIM))
output_matrix = np.random.uniform(-0.5, 0.5, (VOCAB_SIZE, EMBED_DIM))

#training loop
for epoch_idx in range(EPOCHS):
    total_loss_val = 0
    # looping through every word in the data
    for current_pos, target_token_index in enumerate(word_id_list):
        
        #boundries
        start_pos = max(0, current_pos - WINDOW_SIZE)
        end_pos = min(len(word_id_list), current_pos + WINDOW_SIZE + 1)
        
        # loop through the window context
        for context_pos in range(start_pos, end_pos):
            
            # skip the word itself
            if current_pos == context_pos:
                continue 
            
            context_token_index = word_id_list[context_pos]
            #postive matches 
            # forward pass
            raw_score = forwardpress(input_matrix[target_token_index], output_matrix[context_token_index])
            predicted_prob = sigmoid(raw_score)
            
            # track the loss
            total_loss_val += bce_loss(1, predicted_prob)
            
            # backward pass (update with 1)
            # updating the matrix rows in place
            input_matrix[target_token_index], output_matrix[context_token_index] = backpress(
                input_matrix[target_token_index], 
                output_matrix[context_token_index], 
                predicted_prob, 
                1, 
                LEARNING_RATE
            )
            
            ##negative sampling
            # pick 5 random noise words
            for _ in range(5):
                noise_index = np.random.randint(0, VOCAB_SIZE)
                
                # if we picked the real context, skip it
                if noise_index == target_token_index or noise_index == context_token_index:
                    continue 
                
                # forward pass (negative)
                noise_score = forwardpress(input_matrix[target_token_index], output_matrix[noise_index])
                noise_prob = sigmoid(noise_score)
                # track the loss
                total_loss_val += bce_loss(0, noise_prob)
                
                # backward pass, update with 0
                input_matrix[target_token_index], output_matrix[noise_index] = backpress(
                    input_matrix[target_token_index], 
                    output_matrix[noise_index], 
                    noise_prob, 
                    0, 
                    LEARNING_RATE
                )
    # print status every 100 rounds
    if epoch_idx % 100 == 0:
        print(f"Epoch {epoch_idx}: Loss = {total_loss_val:.4f}")
####Test here
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
        
        vec_a = input_matrix[id_a]
        vec_b = output_matrix[id_b]
        
        final_score = np.dot(vec_a, vec_b)
        similarity = sigmoid(final_score)
        print(f"similarity ({word_a} -> {word_b}): {similarity:.4f}"
