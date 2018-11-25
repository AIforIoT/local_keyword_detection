import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

from local_keyword_detection.utils import preprocess

def train(PATH):
    # Import the wav files to mfcc vector
    preprocess.save_data_to_array()
    X_train, X_test, y_train, y_test = preprocess.get_train_test()
    numWords = len(os.listdir(PATH + '/data/'))

    X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
    X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)
    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(numWords, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.Adadelta(), loss='binary_crossentropy', metrics=['accuracy']);

    history = model.fit(X_train, y_train_hot, batch_size=100, epochs=100, verbose=1, validation_data=(X_test, y_test_hot))

    # Serialize model to JSON
    model_json = model.to_json()
    with open(PATH+"/bin/model.json", "w") as json_file:
            json_file.write(model_json)

    # Serialize weights to HDF5
    model.save_weights(PATH+"/bin/model.h5")
    print("Saved model to disk")
