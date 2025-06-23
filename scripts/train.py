import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from diffusion_policy.models.diffusion_policy import DiffusionPolicy
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusion_policy.data.dataset import PushTImageDataset
import typing
import wandb
from diffusion_policy.utils.visualization import visualize_trajectories

dataset_path = '../output/save_data/test_workspace.pkl'
model_path = '../output/diffusion_policy.pth'

obs_horizon = 2  # number of observations to stack
pred_horizon = 16  # number of actions to predict
action_dim = 2  # action dimension, e.g. 2 for push task
action_horizon = 8  # number of actions to output, e.g. 8 for push task
path1 = "../output/save_data/left.pkl"
path2 = "../output/save_data/right.pkl"
def train_diffusion_policy(epochs: int = 100,logging : bool = True):
    #load dataset
    dataset = PushTImageDataset([path1,path2],np.load("../output/save_data/embeddings.npy"), 
                            pred_horizon=16, obs_horizon=2,action_horizon=8)
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
    diffusion_policy = DiffusionPolicy(obs_horizon=obs_horizon, pred_horizon=pred_horizon,
                                       lowdim_obs_dim=2,
                                       action_dim=action_dim,
                                       num_diffusion_iters=100)
    diffusion_policy.to(device)
    #load pretrained weights if available
    # if model_path is not None:
    #     try:
    #         state_dict = torch.load(model_path, map_location=device)
    #         diffusion_policy.load_state_dict(state_dict)
    #         print(f"Loaded pretrained weights from {model_path}")
    #     except FileNotFoundError:
    #         print(f"No pretrained weights found at {model_path}, starting from scratch.")
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
    if logging:
        wandb.init(
            project="Diffusion Policy",  # give your project a name
            name="Policy with simple text conditioning",              # (optional) name of the specific run
            config={
                "epochs": epochs,
                "learning_rate": 1e-4,
                "batch_size": dataloader.batch_size,
                "obs_horizon": diffusion_policy.obs_horizon,
                # add any other hyperparams you want to log
            }
        )
    
    with tqdm(range(epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []

            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # move batch to device
                    nimage = nbatch['image'][:, :obs_horizon].to(device)
                    nagent_pos = nbatch['agent_pos'][:, :obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    ntext = nbatch['text'].to(device)
                    # call forward() to compute loss
                    loss = diffusion_policy(nimage, nagent_pos, naction, ntext)
                    if logging:
                        wandb.log({"loss": loss.item()})
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
            if (epoch_idx+1) % 100 == 0:
                # sample trajectories from the diffusion policy
                nimages = nbatch['image'][0, :obs_horizon].to(device)
                nagent_poses = nbatch['agent_pos'][0, :obs_horizon].to(device)
                ntexts = nbatch['text'][0]  # no slicing needed for text

                naction = diffusion_policy.sample(
                    nimages=nimages,
                    nagent_poses=nagent_poses,
                    ntexts = ntexts,
                    num_diffusion_iters=100,
                    n_samples = 10
                )

                uncond_naction = diffusion_policy.sample(
                    nimages=nimages,
                    nagent_poses=nagent_poses,
                    num_diffusion_iters=100,
                    n_samples = 10
                )
                # unnormalize action
                naction = naction.detach().to('cpu').numpy()
                uncond_naction = uncond_naction.detach().to('cpu').numpy()
                visualize_trajectories(naction, n = 10,gif_path=f"../output/cond_trajectories.gif",)
                visualize_trajectories(uncond_naction, n = 10,gif_path=f"../output/uncond_trajectories.gif",)
    # copy EMA weights into model before saving
    ema.copy_to(diffusion_policy.parameters())

    # save model
    torch.save(diffusion_policy.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    del diffusion_policy, ema, optimizer, lr_scheduler  # free memory
    if logging:
        wandb.finish()  # finish the wandb run

if __name__ == '__main__':
    train_diffusion_policy(epochs=5000,logging = True)  # Adjust epochs as needed
    print("Training complete.")
