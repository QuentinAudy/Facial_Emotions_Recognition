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

#path = "/Users/quentinaudy/PycharmProjects/ferproject/FER/evaluation/test_happy2.jpg"
#path = "/Users/quentinaudy/PycharmProjects/ferproject/Full_process/MTCNN/test.png"


def prediction(image_path):

    image_size=224
    class_names=['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

    test_model = models.load_model('vggface_test.h5')
    test_url= image_path


    img = tf.keras.utils.load_img(test_url, target_size=(image_size, image_size))

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = test_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])


    print(
        "This image most likely belongs to {} with a probability {}."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    return predictions[0]


#prediction(path)


