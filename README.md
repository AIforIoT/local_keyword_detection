# Local Speech to word Neural Net - Tensorflow

This repository contains the source code to build the local Neural Net for the iouti speech recognition.
The code is capable of understanding one wav file including one of the words in /data folder.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
https://blog.manash.me/building-a-dead-simple-word-recognition-engine-using-convnet-in-keras-25e72c19c12b

### Prerequisites

The libraries you must have to run this NN are:

```
Python 3
Tensorflow
h5py
tqdm
librosa
```

### Usage

This package can be used by importing the library:

```
from local_keyword_detection import detect_keyword as dk
```

To detect the wav file if the word is the key word:

```
word = dk.detect(AUDIO_FILE)
```