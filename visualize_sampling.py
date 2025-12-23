import matplotlib.pyplot as plt
import torch
import numpy as np

from ncsn.time_conditioned_mlp import ScoreNetMLP, GaussianFourierProjection
from sampling import sample_2d_data


# Define Sigmas (Geometric Sequence)
sigma_begin = 10.0
sigma_end = 0.01
num_classes = 50 # 50 noise levels
sigmas = torch.tensor(np.geomspace(sigma_begin, sigma_end, num_classes), dtype=torch.float32)

torch.serialization.add_safe_globals([ScoreNetMLP])
torch.serialization.add_safe_globals([GaussianFourierProjection])

model = ScoreNetMLP()
model.load_state_dict(torch.load("TestModel/ncsn/NCSN.pth", weights_only=True))

output = sample_2d_data(model, sigmas)

plt.scatter(output[:,0], output[:,1])
plt.show