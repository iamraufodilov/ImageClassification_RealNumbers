import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


image_height=28
image_width=28
batch_size=5

# loading data from local directory
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'G:/rauf/STEPBYSTEP/Data/numbers/train',
    batch_size=batch_size,
    image_size=(image_height, image_width),
    seed=123)
ds_validate = tf.keras.preprocessing.image_dataset_from_directory(
    'G:/rauf/STEPBYSTEP/Data/numbers/validate',
    batch_size=batch_size,
    image_size=(image_height, image_width),
    seed=123)

# to print out dataset class names
class_names = ds_train.class_names
print(class_names)

# creating the model
num_classes = 10

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(image_height, image_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# get summary from the model
model.summary()

# train the model
epochs=10
model.fit(ds_train,
          validation_data=ds_validate,
          epochs=epochs)

# tomorrow i will continue with prediction

image_path = 'G:/rauf/STEPBYSTEP/Data/numbers/test/4.jpg'
img = keras.preprocessing.image.load_img(image_path, target_size=(image_height, image_width))

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, axis=0)

predicted_data = model.predict(img_array)
score = tf.nn.softmax(predicted_data)
print("this is looks like {}".format(class_names[np.argmax(score)]))
# OMG I finish true first independent prediction with image classification