import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class MixtureOfGaussians(Dataset):
    def __init__(self, n_samples=2000):
        self.n_samples = n_samples
        
        self.mean1 = (2, 3)
        self.mean2 = (1, 1)
        
        self.std1 = 0.5
        self.std2 = 0.3

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        i = np.random.randint(0, 2)
        if i == 0:
            sample = np.random.normal(loc=self.mean1, scale=self.std1)
        else:
            sample = np.random.normal(loc=self.mean2, scale=self.std2)
        return torch.tensor(sample, dtype=torch.float32)

# Quick visualization check
def visualize():
    dset = MixtureOfGaussians(n_samples=2000)
    data = np.array([dset[i].numpy() for i in range(2000)])
    plt.scatter(data[:,0], data[:,1], s=5, alpha=0.6)
    plt.title("Ground Truth: Mixture of Gaussians")
    plt.show()