import os
import torch
from tiny_embodied_reasoning.workspace import utils as utils
import wandb
from transformers import get_cosine_schedule_with_warmup
def trajs_to_tensors(scenes):
    obs_list, act_list = [], []
    for scene in scenes:                     # Scene
        for traj in scene.trajectories:      # Trajectory
            for s in traj.data:               # each s is a State
                obs_list.append(s.observation)
                act_list.append(s.action)
    obs = torch.as_tensor(obs_list, dtype=torch.float32)
    act = torch.as_tensor(act_list, dtype=torch.float32)
    return obs, act

file_path = "../output/save_data/test_workspace.pkl"

from torch.utils.data import Dataset

#normalize the actions to [-1,1]^2 from [0,512]*[0,512]
def normalize_actions(actions):
    actions = actions.float() / 512.0  # Scale to [0, 1]
    actions = actions * 2 - 1           # Scale to [-1, 1]
    return actions

def normalize_observations(observations):
    #normalize observations to [0,1]^3 from [0,255]*[0,255]*[0,255]
    observations = observations.float() / 255.0  # Scale to [0, 1]
    return observations

class DemoDataset(Dataset):
    def __init__(self, obs_tensor: torch.Tensor, act_tensor: torch.Tensor):
        assert len(obs_tensor) == len(act_tensor)
        self.obs, self.act = obs_tensor, act_tensor
        self.obs = obs_tensor.permute(0, 3, 1, 2).contiguous()  # Convert to (B, C, H, W) format if needed
        #normalize actions to [-1, 1]^2
        self.obs = normalize_observations(self.obs)
        self.act = normalize_actions(act_tensor)
        

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx]
from diffusion_policy import DiffusionPolicy

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
import wandb
def train_loop(
    policy: DiffusionPolicy,
    dataset: Dataset,
    epochs: int = 200,
    batch_size: int = 8,
    lr: float = 1e-5,
    warmup_ratio: float = 0.1,
    device: str = "cuda",
):
    """
    Train the diffusion policy model with cosine LR scheduler and linear warmup.
    """
    # Initialize the model
    policy.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    total_steps = epochs * len(loader)
    warmup_steps = int(warmup_ratio * total_steps)

    optim = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    global_step = 0
    ema_loss = None  # initialize outside the training loop

    for epoch in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for obs, act in pbar:
            obs, act = obs.to(device), act.to(device)
            loss = policy(obs, act)

            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            # EMA update
            loss_scalar = loss.item()
            if ema_loss is None:
                ema_loss = loss_scalar
            else:
                ema_loss = 0.01 * loss_scalar + 0.99 * ema_loss

            pbar.set_postfix(ema_loss=f"{ema_loss:.4f}")
            wandb.log({
                "train/loss": loss_scalar,
                "train/ema_loss": ema_loss,
                "train/lr": scheduler.get_last_lr()[0],
                "epoch": epoch,
                "step": global_step,
            })
            global_step += 1
    print(f"Epoch {epoch}: EMA loss={ema_loss:.6f}")


    save_path = f"output/diffusion_policy.pth"
    torch.save(policy.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    wandb.finish()

# Load trajectories from the pickle file
trajs = utils.load_trajectories_pickle(file_path)
obs, act = trajs_to_tensors(trajs)
dataset  = DemoDataset(obs, act)
print(f"obs shape: {obs.shape}, act shape: {act.shape}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
policy = DiffusionPolicy(act_dim=act.shape[1],
                         device=device,schedule = "linear")
policy.load_state_dict(
    torch.load("output/diffusion_policy.pth", map_location=device))
# Initialize wandb for logging
wandb.init(project="Diffusion-Policy", name="train_diffusion_policy",
           config={
               "epochs": 500,
               "batch_size": 8,
               "learning_rate": 1e-5,
               "device": device
           })
train_loop(policy, dataset,
           epochs=500, batch_size=8, device=device,lr = 1e-5)


