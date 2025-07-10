#training script for flow matching policy
import sys
import os
# Add parent directory to path so we can import diffusion_policy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from diffusion_policy.models.flow_matching import FlowMatchingPolicy
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusion_policy.data.dataset import PushTImageDataset
import typing
import wandb
from diffusion_policy.utils.visualization import visualize_trajectories

dataset_path = '../output/save_data/test_workspace.pkl'
model_path = '../trained_models/small_flow_policy_CLIP.pth'

obs_horizon = 2  # number of observations to stack
pred_horizon = 16  # number of actions to predict
action_dim = 2  # action dimension, e.g. 2 for push task
action_horizon = 8  # number of actions to output, e.g. 8 for push task
Colors = ['Blue', 'Red', 'Green']
colors = ['blue', 'red', 'green']
path_list = [f'../dataset/{color}_{num}.pkl' for color in Colors[:1] for num in [0,1,2,3]]
path_list[2] = None
path_list[3] = None
description_list = [f'push the {color} block to the {num} corner' for color in colors[:1] for num in ['lower-right', 'upper-right', 'upper-left', 'lower-left']]
def train_flow_policy(epochs: int = 100,logging : bool = True):
    #load dataset
    dataset = PushTImageDataset(path_list,description_list, 
                            pred_horizon=16, obs_horizon=2,action_horizon=8, rotate = True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=8,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flow_policy = FlowMatchingPolicy(obs_horizon=obs_horizon, pred_horizon=pred_horizon,
                                       lowdim_obs_dim=2,
                                       action_dim=action_dim,
                                       num_diffusion_iters=100,
                                       vision = True,
                                       cached_labels_path = '../output/CLIP_embeddings.pkl')
    flow_policy.to(device)
    #load pretrained weights if available
    # if os.path.exists(model_path):
    #     flow_policy.load_state_dict(torch.load(model_path))
    #     print(f"Loaded pretrained weights from {model_path}")
    # EMA model
    ema = EMAModel(parameters=flow_policy.parameters(), power=0.75)

    # Optimizer
    optimizer = torch.optim.AdamW(
        flow_policy.parameters(), lr=1e-4, weight_decay=1e-6
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
            project="Flow Matching Policy",  # give your project a name
            name="Flow Matching Policy",              # (optional) name of the specific run
            config={
                "epochs": epochs,
                "learning_rate": 1e-4,
                "batch_size": dataloader.batch_size,
                "obs_horizon": flow_policy.obs_horizon,
                # add any other hyperparams you want to log
            }
        )
    #testing if git push works
    try:
        with tqdm(range(epochs), desc='Epoch') as tglobal:
            for epoch_idx in tglobal:
                epoch_loss = []
                with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # move batch to device
                        nimage = nbatch['image'][:, :obs_horizon].to(device)
                        nagent_pos = nbatch['agent_pos'][:, :obs_horizon].to(device)
                        naction = nbatch['action'].to(device)
                        ntext = nbatch['text']
                        # call forward() to compute loss
                        loss = flow_policy(nimage, nagent_pos, naction, ntext)
                        if logging:
                            wandb.log({"loss": loss.item()})
                        # optimize
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()

                        # update EMA
                        ema.step(flow_policy.parameters())

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)

                tglobal.set_postfix(loss=np.mean(epoch_loss))
                if (epoch_idx+1) % 10 == 0:
                    # sample trajectories from the diffusion policy
                    nimages = nbatch['image'][:1, :obs_horizon].to(device)
                    nagent_poses = nbatch['agent_pos'][:1, :obs_horizon].to(device)
                    ntexts = nbatch['text'][:1]  # no slicing needed for text

                    ntexts = [description_list[np.random.randint(0, len(description_list))]]
                    naction = flow_policy.sample(
                        nimages=nimages,
                        nagent_poses=nagent_poses,
                        ntexts = ntexts,
                        nsamples = 10
                    )

                    uncond_naction = flow_policy.sample(
                        nimages=nimages,
                        nagent_poses=nagent_poses,
                        nsamples = 10
                    )
                    # unnormalize action
                    naction = naction.detach().to('cpu').numpy()
                    uncond_naction = uncond_naction.detach().to('cpu').numpy()
                    visualize_trajectories(naction, n = 10,gif_path=f"../output/{ntexts[0]}_trajectories.gif",background_img=nimages[0][0])
                    visualize_trajectories(uncond_naction, n = 10,gif_path=f"../output/uncond_trajectories.gif",background_img=nimages[0][0])
    except KeyboardInterrupt:
        print("Keyboard interrupt, saving model and exiting.")
        # copy EMA weights into model before saving
        ema.copy_to(flow_policy.parameters())
        # save model
        torch.save(flow_policy.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        del flow_policy, ema, optimizer, lr_scheduler  # free memory
    # copy EMA weights into model before saving
    ema.copy_to(flow_policy.parameters())

    # save model
    torch.save(flow_policy.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    del flow_policy, ema, optimizer, lr_scheduler  # free memory
    if logging:
        wandb.finish()  # finish the wandb run

if __name__ == '__main__':
    train_flow_policy(epochs=200,logging = True)  # Adjust epochs as needed
    print("Training complete.")
