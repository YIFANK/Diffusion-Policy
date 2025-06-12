import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from dp import DiffusionPolicy
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from Dataset import PushTImageDataset
import typing

dataset_path = '../output/save_data/test_workspace.pkl'
model_path = '../output/diffusion_policy.pth'

obs_horizon = 2  # number of observations to stack
pred_horizon = 16  # number of actions to predict
action_dim = 2  # action dimension, e.g. 2 for push task
action_horizon = 8  # number of actions to output, e.g. 8 for push task

def train_diffusion_policy(epochs: int = 100):
    #load dataset
    dataset = PushTImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion_policy = DiffusionPolicy(obs_horizon=obs_horizon,
                                       lowdim_obs_dim=2,
                                       action_dim=action_dim,
                                       num_diffusion_iters=100)
    diffusion_policy.to(device)

    # EMA model
    ema = EMAModel(parameters=diffusion_policy.parameters(), power=0.75)

    # Optimizer
    optimizer = torch.optim.AdamW(
        diffusion_policy.parameters(), lr=1e-4, weight_decay=1e-6
    )

    # LR scheduler
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * epochs
    )

    with tqdm(range(epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []

            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # move batch to device
                    nimage = nbatch['image'][:, :diffusion_policy.obs_horizon].to(device)
                    nagent_pos = nbatch['agent_pos'][:, :diffusion_policy.obs_horizon].to(device)
                    naction = nbatch['action'].to(device)

                    # call forward() to compute loss
                    loss = diffusion_policy(nimage, nagent_pos, naction)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                    # update EMA
                    ema.step(diffusion_policy.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            tglobal.set_postfix(loss=np.mean(epoch_loss))

    # copy EMA weights into model before saving
    ema.copy_to(diffusion_policy.parameters())

    # save model
    torch.save(diffusion_policy.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_diffusion_policy(epochs=100)  # Adjust epochs as needed
    print("Training complete.")
