import torch
import torch.nn as nn
import numpy as np
import pickle

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output)
        return output

input_dim = 3
output_dim = 3
hidden_dim = 32

# Generate random input noise 
batch_size = 5
seq_len = 10
features = 3

np.random.seed(10)

noise_cont = np.random.rand(batch_size, seq_len, features).astype(np.float32)
noise_cont = torch.from_numpy(noise_cont)
print(noise_cont.shape)

discrete_values = [1, 2, 3]
noise_disc = np.random.choice(discrete_values, size=(batch_size, seq_len, features), p=[1/3, 1/3, 1/3]).astype(np.float32)
noise_disc = torch.from_numpy(noise_disc)
print(noise_disc.shape)

# Instantiate the generator
generator_cont = Generator(input_dim, hidden_dim, output_dim)
generator_disc = Generator(input_dim, hidden_dim, output_dim)

# Generate synthetic data
synthetic_data_cont = generator_cont(noise_cont)
synthetic_data_disc = generator_disc(noise_disc)

print("Continuous Synthetic Data:", synthetic_data_cont.shape)
cont_file_path = 'synthetic_data_cont.pkl'
with open(cont_file_path, 'wb') as cd:
    pickle.dump(synthetic_data_cont, cd)

print("Discrete Synthetic Data:", synthetic_data_disc.shape)
disc_file_path = 'synthetic_data_disc.pkl'
with open(disc_file_path, 'wb') as dd:
    pickle.dump(synthetic_data_disc, dd)

print("end")