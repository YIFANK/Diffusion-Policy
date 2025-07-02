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
    return [random.randint(95, 105), random.randint(95,105)]
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

@torch.no_grad()
def sample_with_concept_composition(
    diffusion_policy,
    nimages,
    nagent_poses, 
    concept_embeddings,
    concept_weights,
    c0_embedding,
    num_diffusion_iters=100,
    n_samples=1
):
    """
    Sample actions using concept composition during the diffusion process.
    This matches the training logic in compute_concept_loss.
    """
    device = next(diffusion_policy.parameters()).device
    B = n_samples
    
    # Convert numpy arrays to tensors if needed
    if isinstance(concept_embeddings, np.ndarray):
        concept_embeddings = torch.tensor(concept_embeddings, dtype=torch.float32, device=device)
    if isinstance(concept_weights, np.ndarray):
        concept_weights = torch.tensor(concept_weights, dtype=torch.float32, device=device)
    if isinstance(c0_embedding, np.ndarray):
        c0_embedding = torch.tensor(c0_embedding, dtype=torch.float32, device=device)
    
    K = len(concept_embeddings)
    
    # Encode observations (same as in training)
    if diffusion_policy.vision:
        # Repeat for batch dimension
        nimages_batch = nimages.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        nagent_poses_batch = nagent_poses.unsqueeze(0).repeat(B, 1, 1)
        
        image_features = diffusion_policy.vision_encoder(nimages_batch.flatten(end_dim=1))
        image_features = image_features.reshape(B, diffusion_policy.obs_horizon, -1)
        obs_features = torch.cat([image_features, nagent_poses_batch], dim=-1)
    else:
        nagent_poses_batch = nagent_poses.unsqueeze(0).repeat(B, 1, 1)
        obs_features = nagent_poses_batch
    
    obs_cond_base = obs_features.flatten(start_dim=1)
    
    # Initialize action from Gaussian noise
    naction = torch.randn((B, diffusion_policy.pred_horizon, diffusion_policy.action_dim), device=device)
    
    # Set up scheduler
    noise_scheduler = diffusion_policy.noise_scheduler
    noise_scheduler.set_timesteps(num_diffusion_iters)
    
    # DDPM sampling loop with concept composition
    for t in noise_scheduler.timesteps:
        # Base prediction: ε_θ(x_t(τ̃), c_0, s_0, t)
        c0_emb_batch = c0_embedding.unsqueeze(0).repeat(B, 1).to(device)
        obs_cond_c0 = torch.cat([obs_cond_base, c0_emb_batch], dim=-1)
        eps_c0 = diffusion_policy.noise_pred_net(naction, t, global_cond=obs_cond_c0)
        
        # Concept predictions: ε_θ(x_t(τ̃), c̃_k, s_0, t) for each concept k
        eps_pred = eps_c0.clone()
        for k in range(K):
            ck_emb_batch = concept_embeddings[k].unsqueeze(0).repeat(B, 1).to(device)
            obs_cond_ck = torch.cat([obs_cond_base, ck_emb_batch], dim=-1)
            eps_ck = diffusion_policy.noise_pred_net(naction, t, global_cond=obs_cond_ck)
            
            # Add weighted difference: ω_k(ε_θ(x_t(τ̃), c̃_k, s_0, t) - ε_θ(x_t(τ̃), c_0, s_0, t))
            eps_pred += concept_weights[k] * (eps_ck - eps_c0)
        
        # Diffusion step using composed prediction
        naction = noise_scheduler.step(
            model_output=eps_pred,
            timestep=t,
            sample=naction
        ).prev_sample
    
    return naction

def evaluate(max_steps,
            num_episodes = 10,
            model_path: str = '../output/diffusion_policy.pth',
            render: bool = False,
            concept_embeddings = None,
            concept_weights = None):
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
        cfg.agent_info.position = [100,100]
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
            if concept_embeddings is not None and concept_weights is not None:
                # Use concept composition sampling
                c0_embedding = torch.zeros(concept_embeddings.shape[1])
                naction = sample_with_concept_composition(
                    diffusion_policy=diffusion_policy,
                    nimages=nimages,
                    nagent_poses=nagent_poses,
                    concept_embeddings=concept_embeddings,
                    concept_weights=concept_weights,
                    c0_embedding=c0_embedding,
                    num_diffusion_iters=100,
                    n_samples=10
                )
            else:
                # Fall back to original sampling
                naction = diffusion_policy.sample(
                    nimages=nimages,
                    nagent_poses=nagent_poses,
                    num_diffusion_iters=100,
                    n_samples=10
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
        print(f"Episode {episode} score: {sum(rewards)}")
        # if render:
        #     print(f"Saving episode {episode} gif")
        #     images_to_gif(imgs, os.path.join('../output/eval/', f'episode_{episode}.gif'), fps=10)
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
    embeddings = np.load("../output/concepts/bump_embeddings.npy")
    #test the model with the learned concept weights
    learned_weights = np.load("../output/concepts/bump_weights.npy")
    print(f"Loaded embeddings shape: {embeddings.shape}")
    print(f"Loaded weights: {learned_weights}")
    
    # Pass embeddings and weights directly to evaluate function
    # The concept composition happens inside the sampling function
    evaluate(max_steps=50, num_episodes=20, model_path='../output/diffusion_policy.pth', 
             render=True, concept_embeddings=embeddings, concept_weights=learned_weights)
    print("Inference completed.")