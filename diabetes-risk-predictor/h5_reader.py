import h5py
import numpy as np

# Read hdf5 file
hdf = h5py.File('config/model.h5','r')
# Get keys of the hdf5 file
data = list(hdf.keys())
# Read data in each group
for key in data:
    # Get the values
    data = list(hdf[key].values())
    weights = list(data[0]['kernel:0'])
    # Convert to numpy array
    weights = np.array(weights)
    # Print shape of array to see mapping of the layers
    # Format: (input, output) for each layer
    print(weights.shape)