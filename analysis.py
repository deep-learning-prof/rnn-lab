"""
This file is based on the TensorFlow tutorials
"""

"""
### Import TensorFlow
"""
import matplotlib.pyplot as plt
import pandas as pd #we will use this to save the training log

"""
We first load the training log
"""
history = pd.read_csv('rnn-training-history.csv')


"""
### We can see the training accuracy and testing accuracy as follows. 
"""
plt.plot(history['accuracy'], label='accuracy')
plt.plot(history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig('accuracy-plot.png')
plt.clf()


"""
### We can see the training and testing error as follows. 
"""
plt.plot(history['loss'], label='Training Error')
plt.plot(history['val_loss'], label = 'Testing Error')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig('loss-plot.png')
