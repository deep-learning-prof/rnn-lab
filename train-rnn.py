"""
This file is based on the TensorFlow tutorials
"""

"""
This script trains an RNN for text classification """

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
Load our untrained RNN model 
"""
model = tf.keras.models.load_model('untrained-rnn-model')


"""
Training settings. We use default values for binary classification problems
"""
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

"""
Train the model. The validation_steps parameter sets the number of batches in the validatin phase
"""
tic = perf_counter()#start training timer
history = model.fit(train_dataset, 
                    epochs=5,
                    validation_data=test_dataset,
                    validation_steps=30)
toc = perf_counter()# stop training timer
training_time = toc-tic #calculate total training time

"""
Finally, we save the training log, trining time, and the trained model for later analysis
"""
model.save('trained-rnn-model')

history_df = pd.DataFrame(history.history)
history_df.to_csv('rnn-training-history.csv')
np.savetxt('training_time.csv', [training_time], delimiter=',') 
