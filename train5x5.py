import numpy as np
import keras
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras import initializers
from keras import optimizers
from keras.utils import plot_model
from keras.layers import *
from load_data import load_dataset
import pandas as pd

train_x, train_y, test_x, test_y, n_classes, genre = load_dataset(mode="Train", datasetSize=0.75)

# (nx128x128) ==> (nx128x128x1)
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

train_x = train_x / 255.
test_x = test_x / 255.

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=[5,5], kernel_initializer = initializers.he_normal(seed=1), activation="relu", input_shape=(128,128,1)))
# Dim = (124x124x64)
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=[2,2], strides=2))
# Dim = (62x62x64)
model.add(Conv2D(filters=128, kernel_size=[5,5], strides=2, kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
# Dim = (29x29x128)
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=[2,2], strides=2, padding='same'))
# Dim = (15x15x128)
model.add(Conv2D(filters=256, kernel_size=[5,5], kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
# Dim = (11x11x256)
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=[2,2], strides=2, padding='same'))
# Dim = (6x6x256)
model.add(Conv2D(filters=512, kernel_size=[3,3], kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
# Dim = (4x4x512)
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=[2,2], strides=2))
# Dim = (2x2x512)
model.add(BatchNormalization())
model.add(Flatten())
# Dim = (2048)
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Dense(1024, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (1024)
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (256)
model.add(Dropout(0.25))
model.add(Dense(64, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (64)
model.add(Dense(32, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (32)
model.add(Dense(n_classes, activation="softmax", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (8)
print(model.summary())
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])
pd.DataFrame(model.fit(train_x, train_y, epochs=10, verbose=1, validation_split=0.1).history).to_csv("Saved_Model/training_history.csv")
score = model.evaluate(test_x, test_y, verbose=1)
print(score)
model.save("Saved_Model/Model.h5")
