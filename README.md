# Thesis

## EHR-M-GAN to generate synthetic EHR data, focusing on handling missing values 

### Code Explanation:
* data_generation.py : To generate sample discrete and continuous dataset
* disc_data_preprocessing.py : To map alphabetic discrete values to numeric values
* DualVAE.py : Pretraining of dual VAE for continuous and discrete data
* CRN-GANGen.py : CRN generator to give synthetic latent representations
* DualVAE-Gen.py : To create the shared latent space to pass through the decoder 
