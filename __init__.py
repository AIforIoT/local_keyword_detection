from os import path
from local_keyword_detection import train_model

PATH = path.dirname(path.realpath(__file__))

# Check if there is a trained model
if not (path.isfile(PATH+"/bin/model.json") and path.isfile(PATH+"/bin/model.h5")):
    train_model.train(PATH)

