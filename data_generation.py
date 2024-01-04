import numpy as np
import pickle

# Define the number of timelines/batches (n)
num_batches = 5

# Create an empty container to store the timelines
timelines_container = []

# Generate timelines
for _ in range(num_batches):
    # Generate a random timeline length
    timeline_length = 10
    
    # Generate discrete features
    discrete_features = {
        'DiscreteFeature1': np.random.choice(['A', 'B', 'C'], size=timeline_length),
        'DiscreteFeature2': np.random.choice(['X', 'Y', 'Z'], size=timeline_length),
        'DiscreteFeature3': np.random.choice([1, 2, 3], size=timeline_length)
    }

    # Introduce simple rules and randomness to relate the discrete features
    for i in range(1, timeline_length):
        if np.random.rand() < 0.2:
            if discrete_features['DiscreteFeature1'][i - 1] == 'A':
                discrete_features['DiscreteFeature2'][i] = 'X'
            if discrete_features['DiscreteFeature1'][i - 1] == 'C':
                discrete_features['DiscreteFeature3'][i] = 3

    # Generate continuous features
    continuous_features = {
        'ContinuousFeature1': np.random.uniform(0, 1, size=timeline_length),
        'ContinuousFeature2': np.random.uniform(0, 1, size=timeline_length),
        'ContinuousFeature3': np.random.uniform(0, 1, size=timeline_length)
    }

    # Combine discrete and continuous features into a single array
    timeline_array = np.column_stack([discrete_features[key] for key in discrete_features] +
                                     [continuous_features[key] for key in continuous_features])

    # Append the timeline array to the container
    timelines_container.append(timeline_array)

# Convert the container to a NumPy array
timelines = np.array(timelines_container)

# Print the generated timeline data
print((timelines.shape))

# Divide continuous data and discrete data in different arrays
disc_data = timelines[:, :, :3]
cont_data = timelines[:, :, 3:]

print(disc_data.shape)
print(cont_data.shape)

'''print(disc_data)
print(cont_data)'''

# Save the arrays to pickle file
with open('continuous_data.pkl', 'wb') as f:
    pickle.dump(cont_data, f)

with open('discrete_data.pkl', 'wb') as u:
    pickle.dump(disc_data, u)
