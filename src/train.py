"""train script for diffusion policy."""
import argparse
import os
import sys
import warnings
import torch
import diffusion_policy
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from diffusion_policy import DiffusionPolicy

def train_loop(
    policy: DiffusionPolicy,
    dataset: Dataset,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,
    device: str = "cpu",
):
    """
    Train the diffusion policy model based on the provided configuration.
    """
    # Initialize the model
    policy.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optim = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        running_loss = 0.0
        for obs, act in pbar:
            obs, act = obs.to(device), act.to(device)
            loss = policy(obs, act)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item() * obs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        mean_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch}: loss={mean_loss:.6f}")
    # Save the trained model
    model_path = os.path.join("output", "diffusion_policy.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(policy.state_dict(), model_path)
    print(f"Model saved to {model_path}")


    