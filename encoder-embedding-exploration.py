"""
This file is based on the TensorFlow tutorials
"""

"""
This script explores how the encoder works.  """

"""
# Import TensorFlow libraries
"""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

"""
# Download the dataset
"""
dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True)

"""
# Show information about the training dataset. It should show two tensors. One with the reviews and one
with the labels
"""
#print(train_dataset.element_spec)

"""
# Lets print a few examples of the dataset.  
"""
train_examples=dataset['train']

for example_review, example_label in train_examples.take(5):
    print()
    print('text: ', example_review.numpy())
    print('label: ', example_label.numpy())


"""
The vocabulary. When dealing with text, we first need to identify all the words that exist in the
training dataset. The list of unique words is called the vocabulary. We will use this list to replace
words in the training dataset with their index in the vocabulary. TensorFlow provides the vocabulary
building function through its TextVectorization layer. 

In the following, we extract the training samples, and create a TextVectorization
layer called encoder.  
"""

VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)#Initilize layer
encoder.adapt(train_examples.map(lambda text, label: text))#create vocabulary for training dataset

"""
Vocabulary Visualization. Lets take a look at how the vocabulary looks like. You will notice that the
punctuation is gone and that all words are lower case. We do this to simplify the vocabulary. We also 
limited the vocabulary to VOCAB_SIZE so some words will be assigned to UNK (i.e., unknown).
"""
vocab = np.array(encoder.get_vocabulary())
print("These are the first 20 words in the vocabulary:")
print(vocab[:20])


"""
Encoding Visualization. Lets see how some of our reviews will get encoded to vocabulary indexes
"""
encoded_example = encoder(example_review).numpy()
print("Example Review:")
print(example_review)
print("Encoded Example Review")
print(encoded_example)

"""
Embedding. After replaceing words with their vocabulary index, we replace them with a vector that
describes their meaning. Words with similar meanings have vectors with similar values. The idea is that
we are not really interested in the specifc words but rather in their meaning inside the sentences. 
So, before we give the input to the RNN we identify words that are similar. 

To do this we use the Embedding layer which can be trained to find these meaning vectors.

The layer code is below. However, we cannot visualize the vectors until we train the RNN. So for now, we
won't have an output for this part. 
"""

embedding = tf.keras.layers.Embedding(
                                    input_dim=len(encoder.get_vocabulary()),
                                    output_dim=64,
                                    # Use masking to handle the variable sequence lengths
                                    mask_zero=True)




