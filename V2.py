from keras.callbacks import ModelCheckpoint
#import libraries
#import keras
import numpy as np
from keras.applications import vgg16
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
#from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt

#Load the VGG model
image_size=224
#vgg_model = vgg16.VGG16(weights='imagenet')
from keras_vggface.vggface import VGGFace

vgg_conv = VGGFace(weights='vggface', include_top=False, input_shape=(image_size, image_size, 3))

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers:
    layer.trainable = False

from keras import models
from keras import layers
from keras import optimizers

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(7, activation='softmax'))

from keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range=30, # rotate the image 30 degrees
                               width_shift_range=0.1, # Shift the pic width by a max of 10%
                               height_shift_range=0.1, # Shift the pic height by a max of 10%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.2, # Shear means cutting away part of the image (max 20%)
                               zoom_range=0.2, # Zoom in by 20% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )

# Data Generator for Training data
train_generator = image_gen.flow_from_directory('dataset/images/train',
        target_size=(image_size, image_size),
        batch_size=80,
        class_mode='categorical')

validation_generator = image_gen.flow_from_directory('dataset/images/test',
        target_size=(image_size, image_size),
        batch_size=30,
        class_mode='categorical',
        shuffle=False)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=0.001),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=2)

visualizeTheTrainingPerformances(history)