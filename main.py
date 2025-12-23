import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

from time_conditioned_mlp import ScoreNetMLP
from data import MixtureOfGaussians
from training import train
from sampling import sample_2d_data

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ScoreNetMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
dataset = MixtureOfGaussians(n_samples=10000)
modelPath = "TestModel/ncsn/NCSN.pth"

# Define Sigmas (Geometric Sequence)
sigma_begin = 10.0
sigma_end = 0.01
num_classes = 50 # 50 noise levels
sigmas = torch.tensor(np.geomspace(sigma_begin, sigma_end, num_classes), dtype=torch.float32).to(device)

train(model=model, optimizer=optimizer, sigmas=sigmas, num_classes=num_classes, device=device, dset=dataset, modelPath=modelPath)


output = sample_2d_data(model, sigmas, n_samples=1000)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

data = np.array([dataset[i].numpy() for i in range(1000)])
ax1.scatter(data[:,0], data[:,1], s=5, alpha=0.6)
ax1.set_title("True distribution")
ax2.scatter(output[:, 0], output[:, 1], s=5, alpha=0.6)
ax2.set_title("Recreate")
plt.show()