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
import helpers

from keras import backend as keras_backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, SpatialDropout2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical, plot_model
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix


# Define general variables

directory = 'C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/Audio/Audioset/Audioset/Full'

# Set your path to the dataset

models_path = 'C:/Users/simon/Documents/TSP/Cours/HTI/Projet HTI/Audio/models'

# Ensure "channel last" data format on Keras
keras_backend.set_image_data_format('channels_last')

# Define a labels array for future use
labels = [
        'Angry',
        'Disgust',
        'Fear',
        'Happy',
        'Neutral',
        'Sad',
        'Surprised'
    ]

# Pre-processed MEL coefficients
# X_train = np.load("data/X-mel_spec.npy")
# y_train = np.load("data/y-mel_spec.npy")
#
# X_test = np.load("data/X-mel_spec_test.npy")
# y_test = np.load("data/y-mel_spec_test.npy")


X = np.load("data/X-mel_spec_augmented.npy")
y = np.load("data/y-mel_spec_augmented.npy")


print(X.shape)
print(y.shape)
indexes = []

total = 3642
indexes = list(range(0, total))

# Randomize indexes
random.shuffle(indexes)

# Divide the indexes into Train and Test
test_split_pct = 20
split_offset = math.floor(test_split_pct * total / 100)

# Split the metadata
test_split_idx = indexes[0:split_offset]
train_split_idx = indexes[split_offset:total]


# Split the features with the same indexes
X_test = np.take(X, test_split_idx, axis=0)
y_test = np.take(y, test_split_idx, axis=0)
X_train = np.take(X, train_split_idx, axis=0)
y_train = np.take(y, train_split_idx, axis=0)

# Also split metadata

print("X test shape: {} \t X train shape: {}".format(X_test.shape, X_train.shape))
print("y test shape: {} \t\t y train shape: {}".format(y_test.shape, y_train.shape))

le = LabelEncoder()
y_test_encoded = to_categorical(le.fit_transform(y_test))
y_train_encoded = to_categorical(le.fit_transform(y_train))

# How data should be structured
num_rows = 40
num_columns = 272
num_channels = 1

# Reshape to fit the network input (channel last)
print(X_test.shape)
X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns, num_channels)
X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns, num_channels)

# Total number of labels to predict (equal to the network output nodes)
print("Number of labels : ",y_train_encoded.shape[1])
num_labels = y_train_encoded.shape[1]


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

        # Reduces each h×w feature map to a single number by taking the average of all h,w values
        model.add(GlobalAveragePooling2D())

        # Softmax output
        model.add(Dense(num_labels, activation='softmax'))

        return model


# Regularization rates
spatial_dropout_rate_1 = 0.07
spatial_dropout_rate_2 = 0.14
l2_rate = 0.0005

model = create_model(spatial_dropout_rate_1, spatial_dropout_rate_2, l2_rate)


adam = Adam(lr=1e-4, beta_1=0.99, beta_2=0.999)
model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer=adam)

# Display model architecture summary
model.summary()

num_epochs = 500
num_batch_size = 64
model_file = '500Epoch64BatchAugmented.hdf5'
model_path = models_path+'/'+model_file


# # Save checkpoints
# checkpointer = ModelCheckpoint(filepath=model_path,
#                                verbose=1,
#                                save_best_only=True)
# start = datetime.now()
# history = model.fit(X_train,
#                     y_train_encoded,
#                     batch_size=num_batch_size,
#                     epochs=num_epochs,
#                     validation_split=1/12.,
#                     callbacks=[checkpointer],
#                     verbose=1)
#
# duration = datetime.now() - start
# print("Training completed in time: ", duration)

def visualizeTheTrainingPerformances(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.title('Training and testing accuracy')
    plt.plot(epochs, acc, 'bo', label = 'Training accuracy')
    plt.plot(epochs, val_acc, 'b', label = 'Testing accuracy')
    plt.legend()

    plt.figure()
    plt.title('Training and testing loss')
    plt.plot(epochs, loss, 'bo', label = 'Training loss')
    plt.plot(epochs, val_loss, 'b', label = 'Testing loss')
    plt.legend

    plt.show()

    return

# Load best saved model
model = load_model(model_path)

helpers.model_evaluation_report(model, X_train, y_train_encoded, X_test, y_test_encoded)
#visualizeTheTrainingPerformances(history)


# # Predict probabilities for test set
y_probs = model.predict(X_test, verbose=0)

# Get predicted labels
yhat_probs = np.argmax(y_probs, axis=1)
y_trues = np.argmax(y_test_encoded, axis=1)

import importlib
importlib.reload(helpers)

# Sets decimal precision (for printing output only)
np.set_printoptions(precision=2)

# Compute confusion matrix data
cm = confusion_matrix(y_trues, yhat_probs)

helpers.plot_confusion_matrix(cm,
                          labels,
                          normalized=False,
                          title="Model Performance",
                          cmap=plt.cm.Blues,
                          size=(12,12))

