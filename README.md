# Local Speech to word Neural Net - Tensorflow

This repository contains the source code to build the local Neural Net for the speech to word problem.
The code is capable of understanding one wav file including one of the words in /data folder.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The libraries you must have to run this NN are:

```
Python 3
Tensorflow
H5py
tqdm
librosa
```

### Usage

To compile and run the Nerual Network type on your command line:

```
python text_classification_model.py --model_version X /output-dir
```