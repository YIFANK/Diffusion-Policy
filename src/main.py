import os
import torch
from tiny_embodied_reasoning.workspace import utils as utils
import wandb
from transformers import get_cosine_schedule_with_warmup
from visualize_traj import visualize_trajectories
import torch

def trajs_to_tensors(scenes, *, To: int = 2, Tp: int = 8):
    """
    Convert a list of Scene objects → two tensors suitable for the
    (To-obs, Tp-act) diffusion-policy dataset.
    Returns
    -------
    obs_tensor : torch.Tensor  # shape (N, To, …)
    act_tensor : torch.Tensor  # shape (N, Tp, action_dim)
    """
    obs_seq, act_seq = [], []

    for scene in scenes:
        for traj in scene.trajectories:
            states = traj.data
            L = len(states)
            obs_tensor = [torch.as_tensor(states[i].observation) for i in range(L)]
            act_tensor = [torch.as_tensor(states[i].action) for i in range(L)]
            #pad it with the ending state so that the agent learns to stop
            if L < 40:
                obs_tensor += [obs_tensor[-1]] * (40 - L)
                act_tensor += [act_tensor[-1]] * (40 - L)
            L = max(L,40)
            # ----- decide sampling points -------------------------------------
            for mid in range(L):
                # observation ends at mid, action starts at mid
                obs_start = max(0, mid - To + 1)  # inclusive
                obs_end   = mid + 1                # exclusive
                act_start = mid                # inclusive
                act_end   = min(L, mid + Tp)  # exclusive
                obs_block = obs_tensor[obs_start : obs_end]
                act_block = act_tensor[act_start: act_end]
                #padding observation from the left if obs_block is too short
                if len(obs_block) < To:
                    obs_block = [obs_block[0]] * (To - len(obs_block)) + obs_block
                #padding action from the right if act_block is too short
                if len(act_block) < Tp:
                    act_block = act_block + [act_block[-1]] * (Tp - len(act_block))
                # stack along time dimension → (To, …)  and (Tp, action_dim)
                
                obs_seq.append(torch.stack(obs_block, dim=0).permute(0,3,1,2))  # (To, H, W, C) -> (To, C, H, W)
                act_seq.append(torch.stack(act_block, dim=0))

    if not obs_seq:
        raise ValueError("No trajectories long enough for the requested To+Tp!")

    obs_tensor = torch.stack(obs_seq, dim=0)   # (N, To, …)
    act_tensor = torch.stack(act_seq, dim=0)   # (N, Tp, action_dim)
    return obs_tensor, act_tensor


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

class ImageActionDataset(Dataset):
    def __init__(self, trajs, To: int = 1,Tp: int = 8):
        obs_tensor, act_tensor = trajs_to_tensors(trajs, To=To, Tp=Tp)
        assert len(obs_tensor) == len(act_tensor)
        self.To = To
        print(obs_tensor.shape, act_tensor.shape)
        self.obs = normalize_observations(obs_tensor)
        self.act = normalize_actions(act_tensor)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx]

"""train script for diffusion policy."""
import argparse
import os
import sys
import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from dp2 import DiffusionPolicy
import wandb
def train_loop(
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    warmup_ratio: float = 0.1,
    device: str = "cuda",
    n_steps: int = 1000,  # Number of diffusion steps
    schedule: str = "linear",  # "linear" or "cosine"
    To: int = 2,  # number of observations to stack
    Ta: int = 8,  # number of actions to output (action output horizon)
    Tp: int = 8,  # number of actions to predict (action prediction horizon)
    logging: bool = True,
    loading: bool = False,
    model_path: str = "output/diffusion_policy.pth",
    record: bool = False
):
    """
    Train the diffusion policy model with cosine LR scheduler and linear warmup.
    """
    policy = DiffusionPolicy(
        act_dim=2,  # Assuming action dimension is 2
        device=device,
        schedule=schedule,
        n_steps=n_steps,  # Number of diffusion steps
        To=To,
        Tp=Tp
    )
    if loading:
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            policy.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Model file {model_path} does not exist. Starting training from scratch.")
    # Load trajectories from the pickle file
    trajs = utils.load_trajectories_pickle(file_path)
    dataset  = ImageActionDataset(trajs, To=To,Tp = Tp)    
    # Initialize the model
    policy.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    total_steps = epochs * len(loader)
    warmup_steps = int(warmup_ratio * total_steps)

    optim = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    global_step = 0
    ema_loss = None  # initialize outside the training loop
    if logging:
        # Initialize wandb for logging
        wandb.init(project="Diffusion-Policy", name="To-{}-Tp-{}-epochs-{}".format(To, Tp, epochs),
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": 1e-5,
                "schedule": schedule,
                "n_steps": n_steps,
                "To": To,
                "Tp": Tp,
                "device": device
        })
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
            if logging:
                wandb.log({
                    "train/loss": loss_scalar,
                    "train/ema_loss": ema_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": global_step,
                })
            global_step += 1
        if epoch % 10 == 0:
            # save sampled trajectories as checkpoints
            if record:
                obs1 = dataset.obs[0].to(device)
                sampled_trajs = policy.sample_ddim(obs1, n=30,eta = 0.2)
                print(sampled_trajs.mean(), sampled_trajs.std())
                visualize_trajectories(sampled_trajs, n=30, gif_path="../output/ddim_trajectories.gif", fps=5, seed=None)
                sampled_trajs = policy.sample(obs1,n = 30)
                visualize_trajectories(sampled_trajs, n=30, gif_path="../output/ddpm_trajectories.gif", fps=5, seed=None)
            #save the model
            torch.save(policy.state_dict(), model_path)
            print(f"Model saved to {model_path} at epoch {epoch}")
    print(f"Epoch {epoch}: EMA loss={ema_loss:.6f}")
    torch.save(policy.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    if logging:
        wandb.log({"sampled_trajectories": wandb.Video("../output/ddpm_trajectories.gif", caption="Sampled Trajectories")})
    wandb.finish()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# visualize_trajectories(dataset.act, n=100, gif_path="../output/trajectories.gif", fps=5, seed=None)
#visualize trajectories sampled from the model
train_loop(epochs=100, 
    batch_size=2, 
    device=device,
    lr = 1e-4,To = 2,Tp = 16,n_steps = 100,schedule = "linear",
    logging = True,loading = True,
    model_path = "output/diffusion_policy.pth",
    record = True)



