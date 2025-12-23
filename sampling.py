import torch

@torch.no_grad()
def sample_2d_data(model, sigma_levels, n_samples=1000, n_steps_each=100, step_lr=0.00002):
    """
    Specific sampler for the 2D Toy Data.
    """
    device = sigma_levels.device
    model.eval()
    
    # 1. Initialize x with shape (n_samples, 2)
    # Start with random noise scaled by the largest sigma
    x = torch.randn(n_samples, 2, device=device) * sigma_levels[0]
    
    # Iterate through noise levels
    for i, sigma in enumerate(sigma_levels):
        
        # Calculate step size (alpha)
        alpha = step_lr * (sigma / sigma_levels[-1]) ** 2
        
        # Create a tensor of the CURRENT sigma value for the whole batch
        # The MLP expects raw float values, not indices
        current_sigmas = torch.ones(n_samples, device=device) * sigma
        
        for s in range(n_steps_each):
            # Langevin Noise
            z = torch.randn_like(x)
            
            # Predict Score
            # Pass the raw sigma values (floats) to the MLP
            score = model(x, current_sigmas)
            
            # Langevin Update
            noise_term = torch.sqrt(alpha) * z
            score_term = 0.5 * alpha * score
            
            x = x + score_term + noise_term
            
    return x.cpu().numpy()