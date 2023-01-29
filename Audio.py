#imports

import sys
import os
import IPython
import math
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


import random
from datetime import datetime
#rom include import helpers

from keras import backend as keras_backend
from keras.models import Sequential, load_model
from keras.layers import Dense, SpatialDropout2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, LeakyReLU
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical





#Paths and labels

base_directory = './Audioset'
full_directory = base_directory + '/Full'
spectres_directory = base_directory + '/Spectres'


labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprised']




#Create model

def create_model(spatial_dropout_rate_1=0, spatial_dropout_rate_2=0, l2_rate=0):
    # Create a secquential object
    model = Sequential()

    # Conv 1
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     kernel_regularizer=l2(l2_rate),
                     input_shape=(num_rows, num_columns, num_channels)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(SpatialDropout2D(spatial_dropout_rate_1))
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    # Max Pooling #1
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SpatialDropout2D(spatial_dropout_rate_1))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(SpatialDropout2D(spatial_dropout_rate_2))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    # Reduces each h×w feature map to a single number by taking the average of all h,w values.
    model.add(GlobalAveragePooling2D())

    # Softmax output
    model.add(Dense(num_labels, activation='softmax'))

    return model


# Regularization rates
spatial_dropout_rate_1 = 0.07
spatial_dropout_rate_2 = 0.14
l2_rate = 0.0005

model = create_model(spatial_dropout_rate_1, spatial_dropout_rate_2, l2_rate)







#Spectrogram Transformation

for root, dirs, files in os.walk(full_directory):
    for filename in files:
        print(root)
        #print(os.path.join(root, filename))



