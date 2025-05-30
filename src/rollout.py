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
class TestDiffusionPolicy(unittest.TestCase):
    test_env_build_yaml = """
    info:
        render_mode: 'human'
        video_fps: 10
        sim_hz: 100
        control_hz: 10
        render_size: 96
        window_height: 512
        window_width: 512
        seed: 666 # random seed
        window_text: 'Test Build 3'
    
    scene_info: 
        o1:
            geometry: 
                shape: 'box'
                width: 20
                height: 70
            color: 'Brown'
            position: [200, 200] # random position where there is no collision
            angle: 0.2 # random position where there is no collision
        o2:
            geometry:
                shape: 'circle'
                radius: 40
            color: 'Gray'
            position: [220, 220]
    agent_info:
        position: [210, 210]
    """
    # Load policy
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    policy = DiffusionPolicy(act_dim=2, device=device).to(device)
    policy.load_state_dict(
        torch.load("output/diffusion_policy.pth", map_location=device))
    # Load the YAML string into an OmegaConf object
    cfg = OmegaConf.create(test_env_build_yaml)
    pygame.init()
    env = ter_env.TEREnv(**cfg.info, scene_info=cfg.scene_info, agent_info=cfg.agent_info)
    env.reset()
    #image observer
    image_observer = ImageObserver(env, render_size=96, verbose=True)
    quit = False
    action = env.agent.position
    # run the env for 3 control steps
    for i in range(1000):
        # print(env.contacting_set)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True
        if quit:
            break
        observation, reward, done, info = env.step(action)
        image_observation = image_observer.observe()
        action = policy.sample(preprocess(image_observation, policy.device))[0]
        if i % 50 == 0:
            print(action)
        env.render()
        # env.clock.tick(env.control_hz) # run in real time
    env.reset()
    print(env.entities['o2'].position)
    print(env.initial_state['o2']['position'])
    print(env.entities)

if __name__ == '__main__':
    unittest.main()