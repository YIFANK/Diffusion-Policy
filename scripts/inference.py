#@markdown ### **Inference**
import sys
import os
# Add parent directory to path so we can import diffusion_policy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from diffusion_policy.utils.img_to_gif import images_to_gif
import random
import collections
from diffusion_policy.utils.visualization import visualize_trajectories, pad_and_stack_trajectories
from diffusion_policy.data.dataset import PushTImageDataset, normalize_data, unnormalize_data

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
def random_pos():
    # Make sure to keep values within a safe region
    return [random.randint(80, 120), random.randint(80,120)]
def generate_random_scene(seed: int,num_objects: int = 1,rand: bool = False):
    random.seed(seed)
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
def generate_random_position():
    """Generate a random position within the workspace."""
    pos = random_pos()
    return pos

dataset_path = '../output/save_data/test_workspace.pkl'
path1 = "../output/save_data/left.pkl"
path2 = "../output/save_data/right.pkl"
dataset = PushTImageDataset([path1,path2],[-1,1], 
                            pred_horizon=16, obs_horizon=2,action_horizon=8)
stats = dataset.stats
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42

from diffusion_policy.models.diffusion_policy import DiffusionPolicy
def evaluate(max_steps,
            num_episodes = 10,
            model_path: str = '../output/diffusion_policy.pth',
            render: bool = False,
            cond: int = -1):
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
    tot_score = 0
    all_actions = []
    for episode in range(1,num_episodes+1):
        cfg.agent_info.position = generate_random_position()
        env = ter_env.TEREnv(**cfg.info, scene_info=cfg.scene_info, agent_info=cfg.agent_info,
                             verbose = False)
        collision_handler = env.collision_handler
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
            #check for collisions
            if collision_handler.is_colliding("agent"):
                print("Agent is colliding!")
                reward = -1
                done = True
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
        actions = []
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
                ntexts = cond,
                num_diffusion_iters=100,
                n_samples = 10
            )
            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # if render:
            #     visualize_trajectories(
            #         left_naction,
            #         n=10,
            #         gif_path=os.path.join('../output/eval/', f'left_trajectories_{step_idx}.gif'),
            #         fps=10,
            #         seed=seed
            #     )
            #     visualize_trajectories(
            #         right_naction,
            #         n=10,
            #         gif_path=os.path.join('../output/eval/', f'right_trajectories_{step_idx}.gif'),
            #         fps=10,
            #         seed=seed
            #     )
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
                actions.append(normalize_data(action[i],stats = stats['action']))
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
        tot_score += sum(rewards)
        actions = np.stack(actions)
        all_actions.append(actions)
        # save the images as a gif
        if render:
            print(f"Saving episode {episode} gif")
            images_to_gif(imgs, os.path.join('../output/eval/', f'episode_{episode}.gif'), fps=10)
    if render:
        #padding the last action to make all actions same length
        trajs = pad_and_stack_trajectories(all_actions)
        visualize_trajectories(
            trajs,
            n=num_episodes,
            gif_path=os.path.join('../output/eval/', 'all_episodes.gif'),
            fps=10,
            seed=seed
        )
    print(f"Total score: {tot_score} over {num_episodes} episodes")

if __name__ == "__main__":
    # evaluate the model
    embeddings = np.load("../output/save_data/embeddings.npy")
    #test the model with the learned concept weights
    learned_weights = np.load("../output/save_data/learned_concept_weights.npy")
    cond = embeddings[0] * learned_weights[0] + embeddings[1] * learned_weights[1]
    evaluate(max_steps=50, num_episodes=10, model_path='../output/diffusion_policy.pth',render = True,cond = cond)
    print("Inference completed.")