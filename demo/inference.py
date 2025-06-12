#@markdown ### **Inference**
import hydra
from omegaconf import DictConfig, OmegaConf
import unittest
import os
import tiny_embodied_reasoning
import pygame
from tiny_embodied_reasoning.environment import env as ter_env
import numpy as np
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

from dp import DiffusionPolicy
def evaluate(max_steps,
            num_episodes = 10,
            model_path: str = '../output/diffusion_policy.pth'):
    """Evaluate the diffusion policy on the PushTImageEnv."""
    # load the diffusion policy
    diffusion_policy = DiffusionPolicy(
        obs_horizon=obs_horizon,
        num_diffusion_iters=100
    )
    diffusion_policy.to(device)
    # Load the saved weights
    state_dict = torch.load(model_path, map_location=device)
    diffusion_policy.load_state_dict(state_dict)
    # environment setup
    cfg = OmegaConf.create(base_yaml)
    cfg.info.seed = seed
    cfg.scene_info = OmegaConf.create(generate_random_scene(seed))
    cfg.agent_info = OmegaConf.create({
        "position": [210, 210],
    })

    env = ter_env.TEREnv(**cfg.info, scene_info=cfg.scene_info, agent_info=cfg.agent_info)
    tot_score = 0
    for episode in range(num_episodes):
        env.reset()
        image_observer = ImageObserver(env, render_size=96, verbose=False)
        def new_step(action):
            """Step the environment with the given action."""
            obs, reward, done, info = env.step(action)
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
        obs, _, _, _ = new_step(env.agent.position)
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
            nimages = nimages.permute(0, 3, 1, 2)
            # (2,3,96,96)
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)

            # sample actions from the diffusion policy
            naction = diffusion_policy.sample(
                nimages=nimages,
                nagent_poses=nagent_poses,
                num_diffusion_iters=100
            )

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
                obs, reward, done, info = new_step(action[i])
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
        tot_score += max(rewards)
        # save the images as a gif
        images_to_gif(imgs, os.path.join('../output/eval/', f'episode_{episode}.gif'), fps=10)
    print(f"Total score: {tot_score} over {num_episodes} episodes")

if __name__ == "__main__":
    # evaluate the model
    evaluate(max_steps=50, num_episodes=10, model_path='../output/diffusion_policy.pth')
    print("Inference completed.")