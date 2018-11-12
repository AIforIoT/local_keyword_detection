# OS dependencies
import os
import sys

# Tensor flow dependencies
import tensorflow as tf
from tensorflow import keras

import numpy as np
from preprocess import *

# Tensor flow session
sess = tf.Session()

# Load json and create model
json_file = open('bin/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("bin/model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

# Getting the MFCC
sample = wav2mfcc(sys.argv[1])

# We need to reshape it remember?
sample_reshaped = sample.reshape(1, 20, 11, 1)

# Perform forward pass
print(loaded_model.predict(sample_reshaped))
print(get_labels()[0])

# Print the labels, position 0 is the label list. Argmax is the most likely.
print(get_labels()[0][np.argmax(loaded_model.predict(sample_reshaped))])
