# dataset.py

def prepare_text_data(raw_text: str):
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
    
    return word_id_list, vocab_map, reverse_vocab_map