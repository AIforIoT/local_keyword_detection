# System dependencies
import sys
from os import path

# Tensor flow dependencies
import tensorflow as tf
from tensorflow import keras

import numpy as np
from local_keyword_detection.utils import preprocess

PATH = path.dirname(path.realpath(__file__))

# Load json and create model
json_file = open(PATH + '/bin/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights(PATH +"/bin/model.h5")
loaded_model.compile(optimizer=keras.optimizers.Adadelta(), loss='binary_crossentropy', metrics=['accuracy'])

# Flask compatibility
loaded_model._make_predict_function()

def detect(AUDIO_FILE):

    # Getting the MFCC (resampled needed)
    sample = preprocess.wav2mfcc(AUDIO_FILE)
    sample_reshaped = sample.reshape(1, 20, 11, 1)
	
    # Print the probabilites of each word
    #print(loaded_model.predict(sample_reshaped))
    
    keyword = preprocess.get_labels()[0][np.argmax(loaded_model.predict(sample_reshaped))]

    print(keyword)

    if keyword == 'iouti':
        return True
    else:
        return False