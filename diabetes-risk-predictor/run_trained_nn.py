from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import time
import os

# Load JSON
start_time = time.time()
json_file = open('config/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("config/model.h5")
print("Loaded model and weights from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# calculate predictions
test = np.array([[6,148,72,35,0,33.6,0.627,50]])
score = loaded_model.predict(test)
# round predictions
rounded = [round(x[0]) for x in score]
print(rounded[0])
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[0]*100))
print("Execution took {} seconds".format(time.time() - start_time))