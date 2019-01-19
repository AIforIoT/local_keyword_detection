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

# Data for local Speech to word Neural Net

This folder contains the wav files to distinguish the two words our nerual net works on. The folderst contain the following structure:

## Iouti

Recorded by all the members of our organisation: (16 samples)
* Samples 1-5: PC Webcam mic recording
* Samples 6-10: Cellphone headset mic recording.
* Samples 11-13: ESP32 Mic input at 3.3v
* Samples 14-16: ESP32 Mic input at 5v.

### Other

Collection of random words from Google Developer API.
