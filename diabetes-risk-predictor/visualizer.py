from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import time
import os

#Time
start_time = time.time()
# load dataset
np.random.seed(2)
dataset = np.loadtxt("data.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
# create model
model = Sequential()
model.add(Dense(8, activation="relu", input_dim=8, kernel_initializer="normal"))
model.add(Dense(392, activation="relu", kernel_initializer="normal"))
model.add(Dense(392, activation="relu", kernel_initializer="normal"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="normal"))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, epochs=700, batch_size=10, verbose=2)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training'], loc='upper left')
plt.show()