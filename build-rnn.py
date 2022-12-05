"""
This file is based on the TensorFlow tutorials
"""

"""
This script builds an RNN for text classification """

"""
# Import TensorFlow libraries
"""
import numpy as np
import pandas as pd
from time import perf_counter

import tensorflow as tf
import tensorflow_datasets as tfds

"""
# Download the dataset
"""
dataset, info = tfds.load('imdb_reviews', with_info=True,as_supervised=True)
 
"""
# Divide the dataset into training and testing sets
"""
train_dataset, test_dataset = dataset['train'], dataset['test']

"""
Shuffle the data to ensure that we don't train the model with reviews that may realted in someway. 
Recall that we are assuming that each sample is iid. 

We also set the batch size here. This will be used during training
"""
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

"""
Build the encoder
"""
VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))


"""
Build the RNN. Our RNN model will have an encoding layer that replaces the words in our text
with a number, an embedding layer that replaces the number with a vector, a bi-directional
LSTM, a dense layer, and an classification layer.  

"""
model = tf.keras.models.Sequential()
model.add(encoder)
model.add(tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1))

"""
We can verify the model by observing a diagram
Pydot can be installed by running "pip install pydot"
"""
tf.keras.utils.plot_model(model, to_file="rnn_model.png", show_shapes=True)

"""
We can also verify that we are building the model we want by displaying a summary of the layers 
"""
model.summary()

"""
We save the model to train it later
"""
model.save('untrained-rnn-model')


