"""
Program for training the semantic autoencoder for finding the embeddings
"""
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE

from tools import *


# Define the encoder for SAE
# The input should be a batch of MEG measurements and output a batch of word2vec vectors
class Encoder(nn.Module):
    def __init__(self):

        super(Encoder, self).__init__()

        self.in_layer = nn.Linear(n_sensors * n_timepoints, 1000)
        self.hid_layer1 = nn.Linear(1000, 750)
        self.hid_layer2 = nn.Linear(750, 500)
        self.out_layer = nn.Linear(500, word2vec_dim)

    def forward(self, x):

        x = F.relu(self.in_layer(x))
        x = F.relu(self.hid_layer1(x))
        x = F.relu(self.hid_layer2(x))
        x = self.out_layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self):

        super(Decoder, self).__init__()

        self.in_layer = nn.Linear(word2vec_dim, 500)
        self.hid_layer1 = nn.Linear(500, 750)
        self.hid_layer2 = nn.Linear(750, 1000)
        self.out_layer = nn.Linear(1000, n_sensors * n_timepoints)

    def forward(self, z):

        z = F.relu(self.in_layer(z))
        z = F.relu(self.hid_layer1(z))
        z = F.relu(self.hid_layer2(z))
        z = self.out_layer(z)

        return z


skip_training = False  # Set this to true if you're only evaluating a model
generate_dataset = True  # Set to false if dataset already exists
device = torch.device('cpu')

if generate_dataset:
    form_dataset()

X_train, y_train, X_test, y_test = load_dataset()

word2vec_dict = load_word2vec()

# Define the models
encoder = Encoder()
encoder.to(device)

decoder = Decoder()
decoder.to(device)

print("Starting training...\n")

# Train the model
if not skip_training:

    # Values for training size
    dataset_size = 53_000  # Number of datapoints in the complete dataset
    num_epochs = 10  # How many times the whole dataset is passed through the model
    batch_size = 32  # Number of datapoints passed to the model in each iteration
    num_iters = dataset_size // batch_size  # Total number of iterations per epoch

    # Required additional objects
    zs_criterion = nn.MSELoss()
    ul_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

    # Main loop
    for e in range(1, num_epochs + 1):
        loss = -1
        last = 0
        for i in range(num_iters):
            if i % 100 == 0 and i != 0:
                # Print the training result
                print(f"After {i} iterations the loss is: {loss}")

            # Get a new mini-batch
            images, word2vecs, labels, last = next_batch(X_train, y_train, word2vec_dict, batch_size, last)

            # Training
            optimizer.zero_grad()
            vecs = encoder(images)
            out = decoder(vecs)
            loss = ul_criterion(out, images) + zs_criterion(vecs, word2vecs)
            loss.backward()
            optimizer.step()


if not skip_training:
    torch.save(encoder.state_dict(), 'sae_encoder.pth')
    torch.save(decoder.state_dict(), 'sae_decoder.pth')

if skip_training:
    encoder.load_state_dict(torch.load('sae_encoder.pth', map_location=lambda storage, loc: storage))
    encoder.eval()

    decoder.load_state_dict(torch.load('sae_decoder.pth', map_location=lambda storage, loc: storage))
    decoder.eval()

# Pass the training points through the model and save the results

print(f"\nDoing a forward pass on the training points to find the embeddings...\n")

train_embed = []
train_label = []

# Also gather some embeddings for visualizations
labels_to_viz = []
for cat in label_categories:
    labels_to_viz.append(random.choice(label_categories[cat]))

viz_dict = dict(zip(labels_to_viz, [[] for _ in range(len(labels_to_viz))]))

for i in range(X_train.shape[0]):
    point = X_train[i, :]
    label = y_train[i]

    embed = encoder(point)

    if label in labels_to_viz and len(viz_dict[label]) < 100:
        viz_dict[label].append(embed.detach().numpy())

    train_embed.append(embed.detach().numpy())
    train_label.append(label)

# Pass the testing points through the model and save the results

print(f"Doing a forward pass on the testing points to find the embeddings...\n")

test_embed = []
test_label = []

for i in range(X_test.shape[0]):
    point = X_test[i, :]
    label = y_test[i]

    embed = encoder(point)

    test_embed.append(embed.detach().numpy())
    test_label.append(label)

print("Saving the embeddings...")

np.save("train_embeddings.npy", np.array(train_embed))
np.save("train_embedding_labels.npy", np.array(train_label))
np.save("test_embeddings.npy", np.array(test_embed))
np.save("test_embedding_labels.npy", np.array(test_label))


print("Visualizing a sample of the embeddings...")

fig, ax = plt.subplots(1)

all_embeddings = []
for label, embeddings in viz_dict.items():
    all_embeddings.append(np.array(embeddings))

all_embeddings = np.concatenate(all_embeddings)
all_embeddings = TSNE().fit_transform(all_embeddings)

for i, label in enumerate(viz_dict):
    ax.plot(all_embeddings[i * 100:(i + 1) * 100, 0],
            all_embeddings[i * 100:(i + 1) * 100, 1],
            '.', label=label)

plt.legend()
plt.show()

