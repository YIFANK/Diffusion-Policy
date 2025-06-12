#@markdown ### **Inference**
import hydra
from omegaconf import DictConfig, OmegaConf
import unittest
import os
import tiny_embodied_reasoning
import pygame
from tiny_embodied_reasoning.environment import env as ter_env
import numpy as np
from dp2 import DiffusionPolicy
from tiny_embodied_reasoning.observers.observer import StateObserver, ImageObserver
import torch
from img_to_gif import images_to_gif
import random
import collections

obs_horizon = 2  # number of observations to stack
pred_horizon = 16  # number of actions to predict
action_dim = 2  # action dimension, e.g. 2 for push task
action_horizon = 8  # number of actions to output, e.g. 8 for push task

o1 = {
    "geometry": {"shape": "box", "width": 100, "height": 50},
    "color": "Brown",
    "position": [300, 300],  # Random position where there is no collision
    "angle": 0.2,
}
o2 = {
    "geometry": {"shape": "circle", "radius": 40},
    "color": "Gray",
    "position": [250, 250],  # Random position where there is no collision
}
base_yaml = """
    workspace:
        type: 'push'
        save_directory: 'output/save_data'
        save_name: 'test_workspace'
    info:
        render_mode: 'human'
        video_fps: 10
        sim_hz: 100
        control_hz: 10
        render_size: 96
        window_height: 512
        window_width: 512
        seed: 30
        window_text: 'Teleop PicknPlace Workspace + Symbolic Observer'
    scene_info: 
    agent_info: 
        observer:
            type: 'image'
            verbose: True
    """
def generate_random_scene(seed: int,num_objects: int = 1,rand: bool = False):
    random.seed(seed)
    
    def random_pos():
        # Make sure to keep values within a safe region
        return [random.randint(100, 400), random.randint(100,400)]
    if num_objects > 2:
        print("Warning: num_objects > 2, only 2 objects are supported in this example.")
    if num_objects == 1:
        scene = {"o1" : o1}
        if rand:
            scene["o1"]['position'] = random_pos()
    elif num_objects == 2:
        scene = {"o1": o1, "o2": o2}
        if rand:
            scene["o1"]['position'] = random_pos()
            scene["o2"]['position'] = random_pos()
    return scene

from Dataset import PushTImageDataset, normalize_data, unnormalize_data
dataset_path = '../output/save_data/test_workspace.pkl'
dataset = PushTImageDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
stats = dataset.stats
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42

from network import ConditionalUnet1D
from vision_encoder import get_resnet, replace_bn_with_gn
import torch
import torch.nn as nn
vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)

# ResNet18 has output dim of 512
vision_feature_dim = 512
# agent_pos is 2 dimensional
lowdim_obs_dim = 2
# observation feature has 514 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 2
def evaluate(max_steps,
            num_episodes = 10,
            model_path: str = '../output/diffusion_policy.pth'):
    """Evaluate the diffusion policy on the PushTImageEnv."""
    # load the diffusion policy

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    # the final arch has 2 parts
    ema_nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })
    #load the model weights
    ema_nets.load_state_dict(torch.load(model_path, map_location=device))
    ema_nets.eval()
    # environment setup
    cfg = OmegaConf.create(base_yaml)
    cfg.info.seed = seed
    cfg.scene_info = OmegaConf.create(generate_random_scene(seed))
    cfg.agent_info = OmegaConf.create({
        "position": [210, 210],
    })

    env = ter_env.TEREnv(**cfg.info, scene_info=cfg.scene_info, agent_info=cfg.agent_info)

    for i in range(num_episodes):
        env.reset()
        image_observer = ImageObserver(env, render_size=96, verbose=False)
        def new_step(action):
            """Step the environment with the given action."""
            obs, reward, done, info = new_step(action)
            # get image observation
            img = image_observer.observe()
            px,py = env.agent.position
            if px > 450 and py > 450:
                reward = 1
                done = True
            else:
                reward = 0
                done = False
            x = {
                'image': img,
                'agent_pos': env.agent.position
            }
            return x, reward, done, info
        # get first observation
        obs = new_step(env.agent.position)[0]
        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        # save visualization and rewards
        imgs = [obs['image']]
        rewards = list()
        done = False
        step_idx = 0
        while not done:
            B = 1
            # stack the last obs_horizon number of observations
            images = np.stack([x['image'] for x in obs_deque])
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

            # normalize observation
            nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
            # images are already normalized to [0,1]
            nimages = images

            # device transfer
            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
            # (2,3,96,96)
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)

            # infer action
            with torch.no_grad():
                # get image features
                image_features = ema_nets['vision_encoder'](nimages)
                # (2,512)

                # concat with low-dim observations
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_nets['noise_pred_net'](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, _, info = new_step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                imgs.append(obs['image'])

                # update progress bar
                step_idx += 1
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    # print out the maximum target coverage
    print('Score: ', max(rewards))