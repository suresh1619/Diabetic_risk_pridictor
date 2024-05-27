from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import time
import os

#Time
start_time = time.time()
# load dataset
np.random.seed(7)
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
model.fit(X, Y, epochs=680, batch_size=10, verbose=2)
# Evaluate and save the weights to the file
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
print("Execution took {} seconds".format(time.time() - start_time))