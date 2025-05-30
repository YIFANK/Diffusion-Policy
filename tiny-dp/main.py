import torch, random
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
N_TRAJ   = 3_000      # demonstrations
T        = 20         # max steps
OBS, ACT = [], []

for _ in range(N_TRAJ):
    p = torch.rand(2)*2 - 1          # start in [-1,1]^2
    g = torch.rand(2)*2 - 1          # random goal
    for t in range(T):
        obs = p.clone()
        diff = g - p
        if diff.norm() < 0.02: break         # reached
        act = diff.clamp_max(0.1)            # move straight-line, speed ≤0.1
        OBS.append(obs)
        ACT.append(act)
        p += act * 0.05                      # Δt = 0.05

obs_tensor = torch.stack(OBS)                # (N,2)
act_tensor = torch.stack(ACT)                # (N,2)
dataset    = torch.utils.data.TensorDataset(obs_tensor, act_tensor)

from diffusion_policy import DiffusionPolicy
from train import train_loop

device = "cuda" if torch.cuda.is_available() else "cpu"
policy = DiffusionPolicy(obs_dim=2, act_dim=2, n_steps=1000, device= device)
train_loop(policy, dataset, epochs=100, batch_size=256, device=device, lr=1e-4)
#save policy
# Save the trained model
import os
model_path = os.path.join("output", "diffusion_policy.pth")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(policy.state_dict(), model_path)

# Load the trained model
policy = DiffusionPolicy(obs_dim=2, act_dim=2, n_steps=1000, device="cpu")
policy.load_state_dict(torch.load(model_path, map_location))
def rollout_one(policy, start=None, goal=None):
    p = torch.rand(2)*2 - 1 if start is None else start.clone()
    g = torch.rand(2)*2 - 1 if goal  is None else goal.clone()
    traj = [p.tolist()]
    for _ in range(20):
        a = policy.sample(p)[0]
        p += a * 0.05
        traj.append(p.tolist())
        if (p-g).norm() < 0.02: break
    return traj, g.numpy()

traj, goal = rollout_one(policy)
traj = torch.tensor(traj)

plt.scatter(goal[0], goal[1], c='red', label='goal')
plt.plot(traj[:,0], traj[:,1], '-o', label='agent')
plt.legend(); plt.gca().set_aspect('equal'); plt.show()

