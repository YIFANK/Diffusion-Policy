import sys
import os
# Add parent directory to path so we can import diffusion_policy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
import pickle
dataset_path = '../output/save_data/test_workspace.pkl'

obs_horizon = 2  # number of observations to stack
pred_horizon = 16  # number of actions to predict
action_dim = 2  # action dimension, e.g. 2 for push task
action_horizon = 8  # number of actions to output, e.g. 8 for push task
#path to Blue, Red, Green + _ + 0,1,2,3 in a list
Colors = ['Blue', 'Red', 'Green']
colors = ['blue', 'red', 'green']
#only use Blue for now
path_list = [f'../dataset/{color}_{num}.pkl' for color in Colors for num in [0,1,2,3]]
#skip the last two datasets
# path_list[2] = None
# path_list[3] = None
description_list = [f'push the {color} block to the {num} corner' for color in colors for num in ['lower-right', 'upper-right', 'upper-left', 'lower-left']]
def train_diffusion_policy(epochs: int = 200,logging : bool = True,noise_pred_net_type: str = 'transformer',model_path = '../trained_models/blue_diffusion_policy_VLM2Vec.pth',
                           dataset_paths: list = None, descriptions: list = None,
                           vision: bool = True, state_keys: tuple = ('agent',),
                           p_uncond: float = 0.1,
                           early_frames: int = 0, early_weight: float = 5.0,
                           text_lr: float = 1e-2, text_gate: bool = False):
    """dataset_paths/descriptions default to the 12 base goal-conditioned tasks;
    pass explicit lists to train on other splits (e.g. behavior-mode datasets).
    vision=False trains a state-based policy: set state_keys to the entities
    whose positions form the low-dim observation, e.g. ('agent','o1')."""
    if dataset_paths is None:
        dataset_paths = path_list
    if descriptions is None:
        descriptions = description_list
    assert len(dataset_paths) == len(descriptions)
    lowdim_obs_dim = 2 * len(state_keys)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    #load dataset
    with open('../output/cached_labels.pkl', 'rb') as f:
        cached_labels = pickle.load(f)
        for d in descriptions:
            assert d in cached_labels, f"description not in cached_labels: {d}"
    dataset = PushTImageDataset(dataset_paths,descriptions,
                            pred_horizon=16, obs_horizon=2,action_horizon=8,rotate = False,
                            state_keys = state_keys)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        # keep workers low: macOS spawn copies the whole in-memory dataset
        # (~2GB float32 images) into every worker — 8 workers swamped a 16GB machine
        num_workers=2,
        shuffle=True,
        pin_memory=False,  # no effect on MPS
        # don't kill worker process afte each epoch
        persistent_workers=True
    )
    diffusion_policy = DiffusionPolicy(obs_horizon=obs_horizon, pred_horizon=pred_horizon,
                                       lowdim_obs_dim=lowdim_obs_dim,
                                       action_dim=action_dim,
                                       num_diffusion_iters=100,
                                       vision = vision,
                                       cached_labels_path = '../output/cached_labels.pkl',
                                       noise_pred_net_type = noise_pred_net_type,
                                       text_gate = text_gate)
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
    # Optimizer for text encoder and other parameters separately
    # Get the names of parameters from diffusion_policy and text_encoder
    # Get text encoder parameters
    text_encoder_params = list(diffusion_policy.text_encoder.parameters())
    text_encoder_param_ids = set(id(p) for p in text_encoder_params)

    # Get all other parameters that are not in text encoder
    other_params = [p for p in diffusion_policy.parameters() 
                   if id(p) not in text_encoder_param_ids]

    # Set up optimizer with disjoint parameter groups
    # NOTE: text_lr=1e-2 (the historical default) can kill the text pathway:
    # early in training text gradients are noise, and 100x lr + ReLU collapses
    # the projection MLP into a constant function before text becomes useful
    # (observed: pairwise conditional delta == exactly 0 after training).
    optimizer = torch.optim.AdamW(
        [
            {'params': other_params, 'lr': 1e-4},
            {'params': text_encoder_params, 'lr': text_lr}
        ],
        weight_decay=1e-6
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
            name=f"{noise_pred_net_type}-Policy",              # (optional) name of the specific run
            config={
                "epochs": epochs,
                "learning_rate": 1e-4,
                "batch_size": dataloader.batch_size,
                "obs_horizon": diffusion_policy.obs_horizon,
                # add any other hyperparams you want to log
            }
        )
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
                        # decision-frame upweighting: early episode frames are
                        # where the conditioning must break behavioral symmetry
                        sample_weights = None
                        if early_frames > 0:
                            rel = nbatch['rel_frame']
                            sample_weights = 1.0 + (rel < early_frames).float() * (early_weight - 1.0)
                        # call forward() to compute loss
                        loss = diffusion_policy(nimage, nagent_pos, naction, ntext,
                                                p_uncond=p_uncond,
                                                sample_weights=sample_weights)
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
                if epoch_idx % 10 == 0:
                    # sample trajectories from the diffusion policy
                    nimages = nbatch['image'][:1, :obs_horizon].to(device)
                    nagent_poses = nbatch['agent_pos'][:1, :obs_horizon].to(device)
                    #random description
                    ntexts = [descriptions[np.random.randint(0, len(descriptions))]]
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
                    #save model
                    torch.save(diffusion_policy.state_dict(), model_path)
                    print(f"Model saved to {model_path}")
                    # unnormalize action
                    naction = naction.detach().to('cpu').numpy()
                    uncond_naction = uncond_naction.detach().to('cpu').numpy()
                    visualize_trajectories(naction, n = 10,gif_path=f"../output/{ntexts[0]}.gif",background_img=nimage[0][0])
                    visualize_trajectories(uncond_naction, n = 10,gif_path=f"../output/uncond_trajectories.gif",background_img=nimage[0][0])
    except KeyboardInterrupt:
        print("Keyboard interrupt") 
    # copy EMA weights into model before saving
    ema.copy_to(diffusion_policy.parameters())
    # save model
    torch.save(diffusion_policy.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    del diffusion_policy, ema, optimizer, lr_scheduler  # free memory
    if logging:
        wandb.finish()  # finish the wandb run

if __name__ == '__main__':
    train_diffusion_policy(epochs=200,logging = True,noise_pred_net_type = 'unet',model_path = '../trained_models/frozen_resnet.pth')  # Adjust epochs as needed
    print("Training complete.")
