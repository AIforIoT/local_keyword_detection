# System dependencies
import sys, os

# Tensor flow dependencies
import tensorflow as tf
from tensorflow import keras

import numpy as np
from utils import preprocess

def detect(AUDIO_FILE):

	# Tensor flow session
	sess = tf.Session()

	# Load json and create model
	json_file = open('bin/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = keras.models.model_from_json(loaded_model_json)

	# Load weights into new model
	loaded_model.load_weights("bin/model.h5")
	loaded_model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

	# Getting the MFCC (resampled needed)
	sample = preprocess.wav2mfcc(sys.argv[1])
	sample_reshaped = sample.reshape(1, 20, 11, 1)

	# Perform forward pass
	#print(loaded_model.predict(sample_reshaped))
	#print(preprocess.get_labels()[0])

	# Return word detected
	return(preprocess.get_labels()[0][np.argmax(loaded_model.predict(sample_reshaped))])
