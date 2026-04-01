# word2vec
This code uses the Skip-gram model with negative sampling to learn semantic word embedding

Why Skip-gram and not CBOW?

For this model I used a tiny dataset. CBOW averages words together into one vector which averages out the signal and blurs the embedding, This is great for larger datasets as it is not as computationally expensive as Skip-gram however due the size of this particular set, Skip-gram is preferred. 

Skip-gram on the other hand treats every word as a training step so for a window of 2 it does 4 gradient updates per word while CBOW does 1.

On big data Skip-gram is slower however in this case the performance trade off choosing Skip-gram is negligible.

# Justification for architecture choices    
In my model I use Binary cross entropy and also sigmoid.
Why did i not use something like softmax or ReLu?

BCE
Well BCE stands out where 2 inputs are concerned where as softmax is better for cases for where there is more than 2 or thousands

Sigmoid
in this case I needed to know how close they are and needed a % between 0 to 1, sigmoid squishes it down between these numbers while ReLu simply just prevents negative numbers, While the altered version LeakyReLu- instead harshly minimises the negatives instead of just setting it to 0. BCE requires a bound that Relu cannot provide
To start 

# running the project
-> 1. Create and activate the virtual environment
    python3 -m venv .venv
    source .venv/bin/activate  # On macOS/Linux

-> 2. Install dependencies
    pip install -r requirements.txt

-> 3. Run the training engine
    python main.py

# Similar project I worked on but using C from scratch
This task was enjoyable and very reminiscent of another project I worked on, while this project allowed Numpy, I worked on creating a Machine learning library from scratch in C, if you would like to check it out here 

-> https://github.com/M-Ali2404/Digit_Image-detection-from-scratch-using-C