"""
Program for training the semantic autoencoder for finding the embeddings
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from config import *
from tools import *


# Define the encoder for SAE
# The input should be a batch of MEG measurements and output
# a batch of word2vec vectors
class Encoder(nn.Module):
    def __init__(self):

        super(Encoder, self).__init__()

        self.in_layer = nn.Linear(n_sensors * n_timepoints, 1000)
        self.hid_layer = nn.Linear(1000, 500)
        self.out_layer = nn.Linear(500, word2vec_dim)

    def forward(self, x):

        x = F.relu(self.in_layer(x))
        x = F.relu(self.hid_layer(x))
        x = self.out_layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self):

        super(Decoder, self).__init__()

        self.in_layer = nn.Linear(word2vec_dim, 500)
        self.hid_layer = nn.Linear(500, 1000)
        self.out_layer = nn.Linear(1000, n_sensors * n_timepoints)

    def forward(self, z):

        z = F.relu(self.in_layer(z))
        z = F.relu(self.hid_layer(z))
        z = self.out_layer(z)

        return z


skip_training = False  # Set this to true if you're only evaluating a model
device = torch.device('cpu')

# Define the models
encoder = Encoder()
encoder.to(device)

decoder = Decoder()
decoder.to(device)

# Train the model
if not skip_training:

    # Values for training size
    dataset_size = -1  # Number of datapoints in the complete dataset
    num_epochs = 3  # How many times the whole dataset is passed through the model
    batch_size = 32  # Number of datapoints passed to the model in each iteration (defined in trainloader)
    num_iters = dataset_size // batch_size  # Total number of iterations per epoch

    # Required additional objects
    zs_criterion = nn.MSELoss()
    ul_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

    # Main loop
    for e in range(1, num_epochs + 1):
        loss = -1
        for i in range(num_iters):
            # Get a new mini-batch
            images, word2vecs, labels = next_batch(batch_size)

            # Training
            optimizer.zero_grad()
            vecs = encoder(images)
            out = decoder(vecs)
            loss = ul_criterion(out, images) + zs_criterion(vecs, word2vecs)
            loss.backward()
            optimizer.step()

        # Print the training result
        print(f"After {e} epoch(s) the loss is: {loss}")

if not skip_training:
    torch.save(encoder.state_dict(), 'sae_encoder.pth')
    torch.save(decoder.state_dict(), 'sae_decoder.pth')

if skip_training:
    encoder.load_state_dict(torch.load('sae_encoder.pth', map_location=lambda storage, loc: storage))
    encoder.eval()

    decoder.load_state_dict(torch.load('sae_decoder.pth', map_location=lambda storage, loc: storage))
    decoder.eval()

# Pass the training points through the model and save the results
training_points = ...

# Pass the testing points through the model and save the results
testing_points = ...





