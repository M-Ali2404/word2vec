import numpy as np
from dataset import prepare_text_data
from model import SkipGramModel, forwardpress, backpress, sigmoid, bce_loss
from evaluate_model import eval_model

def train():
    # setup the text data
    raw_text = "orange juice king book the of the quick brown fox jumps over the lazy dog"
    
    #calling the functions to get data
    word_id_list, vocab_map, reverse_vocab_map = prepare_text_data(raw_text)

    ##hyperparameters
    VOCAB_SIZE = len(vocab_map)
    EMBED_DIM = 10     # dimension of the vector
    LEARNING_RATE = 0.01
    EPOCHS = 1000      
    WINDOW_SIZE = 2    # look 2 words left and right

    #calling the functions to get data
    model = SkipGramModel(VOCAB_SIZE, EMBED_DIM, LEARNING_RATE)

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
                raw_score = forwardpress(model.input_matrix[target_token_index], model.output_matrix[context_token_index])
                predicted_prob = sigmoid(raw_score)
                
                # track the loss
                total_loss_val += bce_loss(1, predicted_prob)
                
                # backward pass (update with 1)
                # updating the matrix rows in place
                model.input_matrix[target_token_index], model.output_matrix[context_token_index] = backpress(
                    model.input_matrix[target_token_index], 
                    model.output_matrix[context_token_index], 
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
                    noise_score = forwardpress(model.input_matrix[target_token_index], model.output_matrix[noise_index])
                    noise_prob = sigmoid(noise_score)
                    # track the loss
                    total_loss_val += bce_loss(0, noise_prob)
                    
                    # backward pass, update with 0
                    model.input_matrix[target_token_index], model.output_matrix[noise_index] = backpress(
                        model.input_matrix[target_token_index], 
                        model.output_matrix[noise_index], 
                        noise_prob, 
                        0, 
                        LEARNING_RATE
                    )
        # print status every 100 rounds
        if epoch_idx % 100 == 0:
            print(f"Epoch {epoch_idx}: Loss = {total_loss_val:.4f}")
    ####Test here
    eval_model(model, vocab_map)
if __name__ == "__main__":
    train()