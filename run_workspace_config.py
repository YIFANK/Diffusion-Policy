from omegaconf import DictConfig, OmegaConf
import os
import pygame
import tiny_embodied_reasoning as ter
from tiny_embodied_reasoning.workspace import teleop_workspace as tw
import numpy as np

## define the workspace configuration file:

workspace_build_yaml = """
workspace:
    type: 'push'
    'save_directory': 'output/save_data'
    'save_name': 'test_workspace'
info:
    render_mode: 'human'
    video_fps: 10
    sim_hz: 100
    control_hz: 10
    render_size: 96
    window_height: 512
    window_width: 512
    seed: 666 # random seed
    window_text: 'Teleop PicknPlace Workspace + Symbolic Observer'
scene_info: 
    o1:
        geometry: 
            shape: 'circle'
            radius: 20
        color: 'Brown'
        position: [200, 200] # random position where there is no collision
        angle: 0.2 # random position where there is no collision
agent_info: 
    observer:
        type: 'image'
        verbose: True
"""


def main():
    # build the teleop workspace:
    cfg = OmegaConf.create(workspace_build_yaml)
    # build the teleop workspace:
    pygame.init()
    workspace = tw.TeleopWorkspace(cfg)
    workspace.run()
    workspace.close()

if __name__ == "__main__":
    main()