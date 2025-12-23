import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

def train(model, optimizer, sigmas, num_classes, device, dset, modelPath):
    # Training Loop
    dataloader = DataLoader(dset, batch_size=128, shuffle=True)
    epochs = 1000 

    for epoch in range(epochs):
        model.train()
        
        train_loop = tqdm(dataloader, 
                    desc=f"Epoch {epoch + 1}/{epochs} [Training]", 
                    leave=False)
        for x_batch in train_loop:
            x_batch = x_batch.to(device)
            
            # 1. Sample random sigma indices
            indices = torch.randint(0, num_classes, (x_batch.shape[0],), device=device)
            used_sigmas = sigmas[indices]
            
            # 2. Add Noise
            z = torch.randn_like(x_batch)
            x_noisy = x_batch + z * used_sigmas[:, None]
            
            # 3. Predict Score
            # Note: We pass the ACTUAL sigma values, not indices, because our 
            # GaussianFourierProjection expects raw continuous values.
            predicted_score = model(x_noisy, used_sigmas)
            
            # 4. Target Score = -z / sigma
            target_score = -z / used_sigmas[:, None]
            
            # 5. Loss = (score - target)^2 * sigma^2
            loss = torch.sum((predicted_score - target_score)**2, dim=1)
            loss = (loss * (used_sigmas**2)).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1} | Loss: {loss.item():.4f}")
        
    torch.save(model, modelPath)
    print(f"Model saved in: {modelPath}")