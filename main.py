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


image_size=224


def imagePreprocessing(base_directory):
    train_directory = base_directory + '/train'
    test_directory = base_directory + '/test'


    train_datagen = ImageDataGenerator(rescale=1. / 255)
    #train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=5, width_shift_range=0.1,
    #                                   height_shift_range=0.1, zoom_range=0.4, horizontal_flip=False,
    #                                   vertical_flip=False, fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_directory, target_size=(image_size, image_size), batch_size=20,
                                                        class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(test_directory, target_size=(image_size, image_size), batch_size=20,
                                                      class_mode='categorical')


    return train_generator, test_generator




def baselineModel():
    model = keras.models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(512, input_shape=(48, 48, 3), activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(7, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def CNNsimpleModel():
    model = keras.models.Sequential()
    model.add(layers.Conv2D(32, input_shape=(image_size, image_size, 3), kernel_initializer='normal', kernel_size=(3, 3),
                            activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, kernel_initializer='normal', kernel_size=(3, 3),
                            activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(128, kernel_initializer='normal', kernel_size=(3, 3),
                            activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(128, kernel_initializer='normal', kernel_size=(3, 3),
                            activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def defineCNNModelVGGPretrained():


    #baseModel = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    baseModel = VGGFace(weights='vggface',include_top=False, input_shape=(image_size, image_size, 3))

    baseModel.summary()

    for layer in baseModel.layers:
        layer.trainable = False

    for layer in baseModel.layers:
        print(layer, layer.trainable)

    for layer in baseModel.layers[7:18]:
        layer.trainable = True

    VGG_model = models.Sequential()
    VGG_model.add(baseModel)
    VGG_model.add(layers.Flatten())
    VGG_model.add(layers.Dense(512, activation='relu'))
    VGG_model.add(layers.Dropout(rate=0.2))
    VGG_model.add(layers.Dense(7, activation='softmax'))
    VGG_model.compile(loss='categorical_crossentropy', optimizer=rmsprop_v2.RMSprop(learning_rate=0.0001), metrics=['accuracy'])

    return VGG_model


def visualizeTheTrainingPerformances(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    pyplot.title('Training and testing accuracy')
    pyplot.plot(epochs, acc, 'bo', label = 'Training accuracy')
    pyplot.plot(epochs, val_acc, 'b', label = 'Testing accuracy')
    pyplot.legend()

    pyplot.figure()
    pyplot.title('Training and testing loss')
    pyplot.plot(epochs, loss, 'bo', label = 'Training loss')
    pyplot.plot(epochs, val_loss, 'b', label = 'Testing loss')
    pyplot.legend

    pyplot.show()

    return




def main():

    base_directory = "./FER_Extend_2"
    #base_directory = "./dataset/images"
    train_generator, test_generator = imagePreprocessing(base_directory)
    model = defineCNNModelVGGPretrained()
    model.summary()
    history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=50, validation_data=test_generator,
                                  validation_steps=50)
    visualizeTheTrainingPerformances(history)
    return

if __name__ == '__main__':
    main()


