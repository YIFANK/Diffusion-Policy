import os
import torch
from tiny_embodied_reasoning.workspace import utils as utils
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

class DemoDataset(Dataset):
    def __init__(self, obs_tensor: torch.Tensor, act_tensor: torch.Tensor):
        assert len(obs_tensor) == len(act_tensor)
        self.obs, self.act = obs_tensor, act_tensor
        self.obs = obs_tensor.permute(0, 3, 1, 2).contiguous()  # Convert to (B, C, H, W) format if needed

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx]
from diffusion_policy import DiffusionPolicy
from train import train_loop
# Load trajectories from the pickle file
trajs = utils.load_trajectories_pickle(file_path)
obs, act = trajs_to_tensors(trajs)
dataset  = DemoDataset(obs, act)
print(f"obs shape: {obs.shape}, act shape: {act.shape}")
policy = DiffusionPolicy(act_dim=act.shape[1],
                         device="cpu")

train_loop(policy, dataset,
           epochs=50, batch_size=256, device="cpu")


