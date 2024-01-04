import torch
import numpy as np
import pickle
from DualVAE import Decoder

def shared_latent_space_rep(tensor, target_mu, target_logvar):
    mu_tensor = torch.mean(tensor)
    logvar_tensor = torch.std(tensor)

    transformed_tensor = ((tensor - mu_tensor) * (target_logvar / logvar_tensor)) + target_mu

    return transformed_tensor

# Load synthetic data generated from GAN
with open('synthetic_data_cont.pkl', 'rb') as a:
    synthetic_data_cont = pickle.load(a)
with open('synthetic_data_disc.pkl', 'rb') as b:
    synthetic_data_disc = pickle.load(b)

# Load mean from encoder 
with open('output_mean_cont.pkl', 'rb') as c:
    output_mean_cont = pickle.load(c)
with open('output_mean_disc.pkl', 'rb') as d:
    output_mean_disc = pickle.load(d)
    
# Load variance from encoder
with open('output_var_cont.pkl', 'rb') as e:
    output_var_cont = pickle.load(e)
with open('output_var_disc.pkl', 'rb') as f:
    output_var_disc = pickle.load(f)

synthetic_latent_cont = shared_latent_space_rep(synthetic_data_cont, output_mean_cont, output_var_cont)
synthetic_latent_disc = shared_latent_space_rep(synthetic_data_disc, output_mean_disc, output_var_disc)

print(synthetic_latent_cont)
print(synthetic_latent_disc)