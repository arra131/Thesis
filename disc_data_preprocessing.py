import numpy as np
import pickle

# Define a mapping from characters to integers
mapping = {'a': 1, 'b': 2, 'c': 3, 'x': 1, 'y': 2, 'z': 3,}

with open('discrete_data.pkl', 'rb') as f:
    disc_data = pickle.load(f)

# Define the mapping
mapping = {'A': 1, 'B': 2, 'C': 3, 'X': 1, 'Y': 2, 'Z': 3}

# Apply the mapping to the array
for i in range(disc_data.shape[0]):
    for j in range(disc_data.shape[1]):
        for k in range(disc_data.shape[2]):
            disc_data[i, j, k] = mapping.get(disc_data[i, j, k], disc_data[i, j, k])

print(disc_data)

with open('mapped_disc_data.pkl', 'wb') as f:
    pickle.dump(disc_data, f)