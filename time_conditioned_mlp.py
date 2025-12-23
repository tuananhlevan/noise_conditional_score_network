import torch.nn as nn
import torch
import numpy as np

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""
    def __init__(self, embed_dim=256, scale=30.0):
        super().__init__()
        # Randomly initialized weights that are NOT updated during training
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ScoreNetMLP(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Embed noise level (sigma) to a vector
        self.embed = GaussianFourierProjection(embed_dim=hidden_dim)
        
        # Main network layers
        self.linear1 = nn.Linear(2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2) # Output is 2D score vector (dx, dy)
        
        self.act = nn.Softplus() # Smooth activation often works better for scores than ReLU

    def forward(self, x, sigmas):
        # x: (Batch, 2)
        # sigmas: (Batch, ) containing the actual sigma values
        
        # 1. Create time/noise embedding
        # We assume 'sigmas' is the raw sigma value here, not the index
        time_emb = self.embed(sigmas) 
        
        # 2. Inject embedding into hidden layers
        h = self.linear1(x)
        h = h + time_emb # Add time info to features
        h = self.act(h)
        
        h = self.linear2(h)
        h = h + time_emb # Add time info again (skip-like connection)
        h = self.act(h)
        
        h = self.linear3(h)
        h = h + time_emb
        h = self.act(h)
        
        # 3. Output the score
        h = self.output(h)
        
        # Normalize output by sigma (optional, but standard in NCSN++)
        # This helps the network output unit-variance gradients
        return h / sigmas[:, None]