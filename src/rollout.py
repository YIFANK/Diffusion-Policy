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
from visualize_diffusion import visualize_diffusion_process
from visualize_traj import visualize_trajectories

def preprocess(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert HWC uint8 [0‥255] → BCHW float32 [0‥1]."""
    tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    tensor = tensor.permute(2, 0, 1) # BCHW
    return tensor.to(device)

def unnormalize(action: torch.Tensor) -> torch.Tensor:
    """Convert action from [-1, 1]^2 to [0, 512]^2 range."""
    # assert action.dim() == 1 and action.size(0) == 2, "Action must be a 2D tensor"
    #clip action to [-1, 1] range
    action = torch.clamp(action, -1, 1)
    return (action + 1) * 256  # Scale to [0, 512] range

def normalize_observation(obs: torch.Tensor) -> torch.Tensor:
    """Convert observation from [0, 1]^3 to [0, 255]^3 range."""
    #print(f"Observation shape: {obs.shape}")
    #clip observation to [-1, 1] range
    return obs / 255.0  # Scale to [0, 1] range

import random
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

def test_diffusion_policy(
    policy_path: str,
    To: int = 2,
    Tp: int = 8,
    Ta: int = 8,
    n_steps: int = 1000,
    schedule: str = "linear",
    num_episodes: int = 10,
    steps_per_episode: int = 20,
    render: bool = True,
    save_path: str = '../output',
    image_save: bool = False,
    rand: bool = False,
):
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    policy = DiffusionPolicy(act_dim=2, To = To, Tp = Tp,n_steps=n_steps, schedule = schedule, device=device).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

    pygame.init()

    def run_single_episode(seed: int):
        cfg = OmegaConf.create(base_yaml)
        cfg.info.seed = seed
        cfg.scene_info = OmegaConf.create(generate_random_scene(seed))
        cfg.agent_info = OmegaConf.create({
            "position": [210, 210],
        })

        env = ter_env.TEREnv(**cfg.info, scene_info=cfg.scene_info, agent_info=cfg.agent_info)
        env.reset()
        image_observer = ImageObserver(env, render_size=96, verbose=False)

        obs_stack = []
        act_stack = []
        observations = []
        for i in range(steps_per_episode):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            action = env.agent.position
            if act_stack:
                action = act_stack.pop(0)
            observation, reward, done, info = env.step(action)
            image_observation = image_observer.observe()
            observations.append(image_observation)
            img = preprocess(image_observation, device)
            while len(obs_stack) <= To:
                obs_stack.append(img)
            if len(obs_stack) > To:
                obs_stack.pop(0)

            if len(obs_stack) == To and len(act_stack) == 0:
                obs_tensor = torch.stack(obs_stack, dim=0) # Shape: ( To, 3, H, W)
                # print(f"Observation tensor shape: {obs_tensor.shape}")
                obs_tensor = normalize_observation(obs_tensor)
                action = policy.sample(obs_tensor,n = 5)
                # action = policy.sample_ddim(obs_tensor, n = 5,steps = 50,eta = 0.1)
                # visualize_trajectories(action, n=5, gif_path=f"../output/episode_{seed}_traj_{i}.gif", fps=5, seed=None)
                action = unnormalize(action[0])
                # print(f"Action shape after sampling: {action.shape}")
                #action.shape = (Tp, 2), append each action in action to act_stack
                for j in range(Ta):
                    #only append the first Ta actions
                    act_stack.append(action[j])
                #visualization
                if i % 20 == 0:
                    visualize_diffusion_process(policy, obs_tensor, n=10, stride=10, gif_path= f"../output/denoise_{i}.gif", dpi=120)

            if i % 5 == 0:
                print(f"Seed {seed} | Step {i} | Action: {action}")
            if render:
                env.render()
        #save the observations as a gif
        if image_save:
            images_to_gif(observations, os.path.join(save_path, f"episode_{seed}.gif"), fps=10)
        env.reset()

    for ep in range(num_episodes):
        seed = ep
        if rand:
            seed = random.randint(0, 1000)
        print(f"\n--- Running Episode {ep+1} with Seed {seed} ---")
        run_single_episode(seed)

if __name__ == '__main__':
    test_diffusion_policy(
        policy_path='output/diffusion_policy.pth',
        To = 2,
        Tp = 8,
        Ta = 4,
        n_steps = 100,
        num_episodes=1,
        steps_per_episode=30,
        schedule = 'sigmoid',
        render=True,
        image_save=True,
    )