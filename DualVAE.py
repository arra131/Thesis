import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np

class SharedLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SharedLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]

class Encoder(nn.Module):
    def __init__(self, shared_layer, latent_size):
        super(Encoder, self).__init__()
        self.fc_mean = nn.Linear(shared_layer.lstm.hidden_size, latent_size)
        self.fc_logvar = nn.Linear(shared_layer.lstm.hidden_size, latent_size)
        self.shared_layer = shared_layer

    def forward(self, x):
        x = self.shared_layer(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean.unsqueeze(2), logvar.unsqueeze(2)

class Decoder(nn.Module):
    def __init__(self, shared_layer, latent_size, output_size, seq_len):
        super(Decoder, self).__init__()
        self.shared_layer = shared_layer
        self.fc_latent = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc_output = nn.Linear(hidden_size, output_size)
        self.seq_len = seq_len

    def forward(self, z, seq_len):
        #zu = z.unsqueeze(1).repeat(1, seq_len, 1)
        z = self.fc_latent(z.squeeze(2))
        z = F.relu(z)
        z = z.unsqueeze(1)
        _, (h_n, c_n) = self.decoder_lstm(z)
        reconstructed = self.fc_output(h_n[-1, :])
        return reconstructed

class DualVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size):
        super(DualVAE, self).__init__()
        self.shared_layer = SharedLayer(input_size, hidden_size)
        self.encoder_1 = Encoder(self.shared_layer, latent_size)
        self.encoder_2 = Encoder(self.shared_layer, latent_size)
        self.decoder_1 = Decoder(self.shared_layer, latent_size, output_size, seq_len)
        self.decoder_2 = Decoder(self.shared_layer, latent_size, output_size, seq_len)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, continuous_data, discrete_data):
        output_mean_1, output_logvar_1 = self.encoder_1(continuous_data)
        output_mean_2, output_logvar_2 = self.encoder_2(discrete_data)

        z_1 = self.reparameterize(output_mean_1, output_logvar_1)
        z_2 = self.reparameterize(output_mean_2, output_logvar_2)

        decoded_output_1 = self.decoder_1(z_1, seq_len)
        decoded_output_2 = self.decoder_2(z_2, seq_len)

        return decoded_output_1, decoded_output_2

# Implement other losses later
def loss_function (recon_x, x, mean, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return KLD

# Load the data
with open('continuous_data.pkl', 'rb') as f:
    cont_data = pickle.load(f)
cont_data = cont_data.astype(np.float32)

with open('mapped_disc_data.pkl', 'rb') as g:
    disc_data = pickle.load(g)
disc_data = disc_data.astype(np.float32)

# Convert data to a tensor
continuous_data = torch.from_numpy(cont_data)
discrete_data = torch.from_numpy(disc_data)

print("Shape of continuous input data is", continuous_data.shape)
print("Shape of discrete input data is", discrete_data.shape)

input_size = 3  # Replace with your actual input size
hidden_size = 20  # Replace with your desired hidden size
latent_size = 20  # Replace with your desired latent size
output_size = 3
seq_len = 10

dual_vae = DualVAE(input_size, hidden_size, latent_size, output_size)
optimizer = optim.Adam(dual_vae.parameters(), lr=0.001)

epochs = 100 
for epoch in range(epochs):
    dual_vae.train()
    optimizer.zero_grad()

    decoded_output_1, decoded_output_2 = dual_vae(continuous_data, discrete_data)
    output_mean_1, output_logvar_1 = dual_vae.encoder_1(continuous_data)
    output_mean_2, output_logvar_2 = dual_vae.encoder_2(discrete_data)
    decoded_output_1 = decoded_output_1.unsqueeze(0)
    decoded_output_2 = decoded_output_2.unsqueeze(0)

    # loss calculation 
    loss_1 = loss_function(decoded_output_1, continuous_data, output_mean_1, output_logvar_1)
    loss_2 = loss_function(decoded_output_2, discrete_data, output_mean_2, output_logvar_2)

    loss_total = loss_1 + loss_2

    loss_total.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss_total.item() / (len(continuous_data) * seq_len)}')

# Weight sharing verification
print("Weight sharing status between encoders:", dual_vae.encoder_1.shared_layer.lstm.weight_ih_l0 is dual_vae.encoder_2.shared_layer.lstm.weight_ih_l0)
print("Weight sharing status between decoders:", dual_vae.decoder_1.shared_layer.lstm.weight_ih_l0 is dual_vae.decoder_2.shared_layer.lstm.weight_ih_l0)

print(decoded_output_1.shape)
print(decoded_output_2.shape)

# Used for creating shared latent space
cont_mean_file_path = 'output_mean_cont.pkl'
with open(cont_mean_file_path, 'wb') as cm:
    pickle.dump(output_mean_1, cm)

disc_mean_file_path = 'output_mean_disc.pkl'
with open(disc_mean_file_path, 'wb') as dm:
    pickle.dump(output_mean_2, dm)   

cont_var_file_path = 'output_var_cont.pkl'
with open(cont_var_file_path, 'wb') as cv:
    pickle.dump(output_logvar_1, cv)

disc_var_file_path = 'output_var_disc.pkl'
with open(disc_var_file_path, 'wb') as dv:
    pickle.dump(output_logvar_2, dv) 