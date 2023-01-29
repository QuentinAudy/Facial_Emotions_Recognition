import cv2 as cv
import numpy as np
import keras
import os
import shutil
from matplotlib import pyplot
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras.optimizers import rmsprop_v2

from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.mobilenet import MobileNet
from keras_vggface.vggface import VGGFace

import librosa
import librosa.display
import helpers

#path = "/Users/quentinaudy/PycharmProjects/ferproject/test.wav"


def predictionaudio(image_path):

    #image_size=224
    class_names=['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

    # Extract Log-Mel Spectrograms (do not add padding)
    mels = helpers.get_mel_spectrogram(image_path, 0, n_mels=40)
    features = []
    frames_max = 272

    # Save current frame count
    num_frames = mels.shape[1]
    features.append(mels)

    padded_features = helpers.add_padding(features, frames_max)
    X = np.array(padded_features)
    print(X.shape)

    test_model = models.load_model('200Epoch64BatchAugmented.hdf5')
    #test_url= image_path


    #img = tf.keras.utils.load_img(test_url, target_size=(image_size, image_size))

    #img_array = tf.keras.utils.img_to_array(img)
    #img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = test_model.predict(X)
    score = tf.nn.softmax(predictions[0])


    print(
        "This audio most likely belongs to {} with a probability {}."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    return class_names[np.argmax(score)], 100 * np.max(score)


#predictionaudio(path)


