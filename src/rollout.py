import hydra
from omegaconf import DictConfig, OmegaConf
import unittest
import os
import tiny_embodied_reasoning
import pygame
from tiny_embodied_reasoning.environment import env as ter_env
import numpy as np
from diffusion_policy import DiffusionPolicy
from tiny_embodied_reasoning.observers.observer import StateObserver, ImageObserver
import torch
def preprocess(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert HWC uint8 [0‥255] → BCHW float32 [0‥1]."""
    tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # BCHW
    return tensor.to(device)

def unnormalize(action: torch.Tensor) -> torch.Tensor:
    """Convert action from [-1, 1]^2 to [0, 512]^2 range."""
    assert action.dim() == 1 and action.size(0) == 2, "Action must be a 2D tensor"
    #clip action to [-1, 1] range
    action = torch.clamp(action, -1, 1)
    return (action + 1) * 256  # Scale to [0, 512] range

def unnormalize_observation(obs: torch.Tensor) -> torch.Tensor:
    """Convert observation from [0, 1]^3 to [0, 255]^3 range."""
    #print(f"Observation shape: {obs.shape}")
    #clip observation to [-1, 1] range
    obs = torch.clamp(obs, 0, 1)
    return obs * 255  # Scale to [0, 255] range

import random

def generate_random_scene(seed: int):
    random.seed(seed)
    
    def random_pos():
        # Make sure to keep values within a safe region
        return [random.randint(0, 512), random.randint(0,512)]

    return {
        "o1": {
            "geometry": {"shape": "circle", "radius": 80},
            "color": "Blue",
            "position": random_pos(),
        },
        "o2": {
            "geometry": {"shape": "box", "width": 50, "height": 150},
            "color": "Brown",
            "position": random_pos(),
            "angle": random.uniform(0, 3.1416),
        },
    }
class TestDiffusionPolicy(unittest.TestCase):
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
        o1:
            geometry:
                shape: 'circle'
                radius: 50
            color: 'Purple'
            position: [300, 400]
        o2:
            geometry: 
                shape: 'box'
                width: 100
                height: 50
            color: 'Purple'
            position: [300, 300]
            angle: 0
    agent_info: 
        observer:
            type: 'image'
            verbose: True
    """

    @classmethod
    def setUpClass(cls):
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {cls.device}")
        # cls.policy = DiffusionPolicy(act_dim=2, device=cls.device).to(cls.device)
        # cls.policy.load_state_dict(torch.load("output/new_diffusion_policy.pth", map_location=cls.device))
        cls.policy = DiffusionPolicy(act_dim=2, device=cls.device,beta_end = 0.005,n_steps = 2000).to(cls.device)
        cls.policy.load_state_dict(torch.load("output/diffusion_policy_more_steps.pth", map_location=cls.device))
        pygame.init()

    def run_agent_with_seed(self, seed: int):
        # Load config and update seed
        cfg = OmegaConf.create(self.base_yaml)
        cfg.info.seed = seed
        cfg.scene_info = OmegaConf.create(generate_random_scene(seed))
        cfg.agent_info = OmegaConf.create({
            "position": [random.randint(0, 512), random.randint(0, 512)]
        })
        # Setup environment
        env = ter_env.TEREnv(**cfg.info, scene_info=cfg.scene_info, agent_info=cfg.agent_info)
        env.reset()
        image_observer = ImageObserver(env, render_size=96, verbose=True)

        quit = False
        action = env.agent.position
        for i in range(10):  # steps per episode
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit = True
            if quit:
                break

            observation, reward, done, info = env.step(action)
            image_observation = image_observer.observe()
            obs = unnormalize_observation(preprocess(image_observation, self.device))
            action = self.policy.sample(obs)[0]
            action = unnormalize(action)

            if i % 5 == 0:
                print(f"Seed {seed} | Step {i} | Action: {action}")
            env.render()
        env.reset()

    def test_diffusion_policy_seeds(self):
        for _ in range(10): # Number of seeds to test
            seed = random.randint(0, 1000)
            with self.subTest(seed=seed):
                print(f"\n--- Running agent with seed {seed} ---")
                self.run_agent_with_seed(seed)

if __name__ == '__main__':
    unittest.main()