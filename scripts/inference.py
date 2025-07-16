#@markdown ### **Inference**
import sys
print(sys.path)
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
import pickle
obs_horizon = 2  # number of observations to stack
pred_horizon = 16  # number of actions to predict
action_dim = 2  # action dimension, e.g. 2 for push task
action_horizon = 8  # number of actions to output, e.g. 8 for push task

o1 = {
    "geometry": {"shape": "box", "width": 100, "height": 100},
    "color": "Brown",
    "position": [300, 300],  # Random position where there is no collision
    "angle": 0.2,
}
o2 = {
    "geometry": {"shape": "circle", "radius": 40},
    "color": "Gray",
    "position": [250, 250],  # Random position where there is no collision
}
green_yaml = """
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
        seed: 666 # random seed
        window_text: 'Teleop PicknPlace Workspace + Symbolic Observer'
    scene_info:  
        o1:
            geometry: 
                shape: 'box'
                width: 100
                height: 100
            color: 'Green'
            # position: [250, 250] # random position where there is no collision
            angle: 0
        # o2: 
        #     geometry:
        #         shape: 'box'
        #         width: 100
        #         height: 100
        #     color: 'Yellow'
        #     # position: [100,100]
        #     angle: 0
    agent_info: 
        # position: [50, 50]
        observer:
            type: 'image'
            verbose: True
    """
blue_yaml = green_yaml.replace('Green', 'Blue')
red_yaml = green_yaml.replace('Green', 'Red')
yellow_yaml = green_yaml.replace('Green', 'Yellow')
yamls = [blue_yaml, red_yaml, green_yaml, yellow_yaml]
task_names = ['blue', 'red', 'green', 'yellow']
win_condition = [
    lambda x: x['o1']['position'][0] > 416 and x['o1']['position'][1] < 96, #lower-right
    lambda x: x['o1']['position'][0] > 416 and x['o1']['position'][1] > 416, #upper-right
    lambda x: x['o1']['position'][0] < 96 and x['o1']['position'][1] > 416, #upper-left
    lambda x: x['o1']['position'][0] < 96 and x['o1']['position'][1] < 96, #lower-left
]
corner_names = ['lower-right', 'upper-right', 'upper-left', 'lower-left']
dataset_path = '../output/save_data/test_workspace.pkl'
path_list = [f'../dataset/{color}_{num}.pkl' for color in ['Blue', 'Red', 'Green'] for num in [0,1,2,3]]
description_list = [f'push the {color} block to the {corner} corner' for color in ['blue', 'red', 'green'] for corner in corner_names]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42

from diffusion_policy.models.diffusion_policy import DiffusionPolicy
from diffusion_policy.models.flow_matching import FlowMatchingPolicy

@torch.no_grad()
def sample_with_concept_composition(
    policy,
    nimages,
    nagent_poses, 
    concept_embeddings,
    concept_weights,
    c0_embedding,
    num_diffusion_iters=100,
    n_samples=1,
    policy_type="diffusion",
):
    """
    Sample actions using concept composition during the sampling process.
    Supports both diffusion and flow matching policies.
    This matches the training logic in compute_concept_loss.
    """
    device = next(policy.parameters()).device
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
    if policy.vision:
        # Repeat for batch dimension
        nimages_batch = nimages.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        nagent_poses_batch = nagent_poses.unsqueeze(0).repeat(B, 1, 1)
        
        image_features = policy.vision_encoder(nimages_batch.flatten(end_dim=1))
        image_features = image_features.reshape(B, policy.obs_horizon, -1)
        obs_features = torch.cat([image_features, nagent_poses_batch], dim=-1)
    else:
        nagent_poses_batch = nagent_poses.unsqueeze(0).repeat(B, 1, 1)
        obs_features = nagent_poses_batch
    
    obs_cond_base = obs_features.flatten(start_dim=1)
    
    # Initialize action from Gaussian noise
    naction = torch.randn((B, policy.pred_horizon, policy.action_dim), device=device)
    
    if policy_type == "diffusion":
        # DDPM sampling loop with concept composition
        noise_scheduler = policy.noise_scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)
        
        for t in noise_scheduler.timesteps:
            # Base prediction: ε_θ(x_t(τ̃), c_0, s_0, t)
            c0_emb_batch = c0_embedding.unsqueeze(0).repeat(B, 1).to(device)
            obs_cond_c0 = torch.cat([obs_cond_base, c0_emb_batch], dim=-1)
            eps_c0 = policy.noise_pred_net(naction, t, global_cond=obs_cond_c0)
            
            # Concept predictions: ε_θ(x_t(τ̃), c̃_k, s_0, t) for each concept k
            eps_pred = eps_c0.clone()
            for k in range(K):
                ck_emb_batch = concept_embeddings[k].unsqueeze(0).repeat(B, 1).to(device)
                obs_cond_ck = torch.cat([obs_cond_base, ck_emb_batch], dim=-1)
                eps_ck = policy.noise_pred_net(naction, t, global_cond=obs_cond_ck)
                
                # Add weighted difference: ω_k(ε_θ(x_t(τ̃), c̃_k, s_0, t) - ε_θ(x_t(τ̃), c_0, s_0, t))
                eps_pred += concept_weights[k] * (eps_ck - eps_c0)
            
            # Diffusion step using composed prediction
            naction = noise_scheduler.step(
                model_output=eps_pred,
                timestep=t,
                sample=naction
            ).prev_sample
    
    elif policy_type == "flow_matching":
        # Flow matching sampling with concept composition
        dt = 1.0 / num_diffusion_iters
        
        for i in range(num_diffusion_iters):
            t_val = i * dt
            timestep_scaled = t_val * 100  # Scale to 0-100 range
            
            # Base prediction: v_θ(x_t(τ̃), c_0, s_0, t)
            c0_emb_batch = c0_embedding.unsqueeze(0).repeat(B, 1).to(device)
            obs_cond_c0 = torch.cat([obs_cond_base, c0_emb_batch], dim=-1)
            v_c0 = policy.flow_predictor(naction, timestep=timestep_scaled, global_cond=obs_cond_c0)
            
            # Concept predictions: v_θ(x_t(τ̃), c̃_k, s_0, t) for each concept k
            v_pred = v_c0.clone()
            for k in range(K):
                ck_emb_batch = concept_embeddings[k].unsqueeze(0).repeat(B, 1).to(device)
                obs_cond_ck = torch.cat([obs_cond_base, ck_emb_batch], dim=-1)
                v_ck = policy.flow_predictor(naction, timestep=timestep_scaled, global_cond=obs_cond_ck)
                
                # Add weighted difference: ω_k(v_θ(x_t(τ̃), c̃_k, s_0, t) - v_θ(x_t(τ̃), c_0, s_0, t))
                v_pred += concept_weights[k] * (v_ck - v_c0)
            
            # Euler step: x_{t+dt} = x_t + dt * v_t
            naction = naction + dt * v_pred
    
    else:
        raise ValueError(f"Unsupported policy_type: {policy_type}. Must be 'diffusion' or 'flow_matching'")
    
    return naction

def evaluate(max_steps = 200,
            num_episodes = 10,
            model_path: str = '../output/diffusion_policy.pth',
            render: bool = True,
            concept_embeddings = None,
            concept_weights = None,
            policy_type = "diffusion",
            task = ['blue', 1],
            init_type = 'original',
            noise_pred_net_type = 'unet'):
    """Evaluate the policy (diffusion or flow matching) on the PushTImageEnv."""
    #model name is the model path without the extension
    model_name = model_path.split('/')[-1].split('.')[0]
    print(f"Model name: {model_name}")
    dataset = PushTImageDataset(path_list,description_list, 
                            pred_horizon=16, obs_horizon=2,action_horizon=8)
    stats = dataset.stats
    # load the policy
    if policy_type == "diffusion":
        policy = DiffusionPolicy(
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            lowdim_obs_dim=2,
            action_dim=action_dim,
            num_diffusion_iters=100,
            vision=True,
            text=True,
            cached_labels_path = '../output/cached_labels.pkl',
            noise_pred_net_type = noise_pred_net_type
        )
    elif policy_type == "flow_matching":
        policy = FlowMatchingPolicy(obs_horizon=obs_horizon, pred_horizon=pred_horizon,
                lowdim_obs_dim=2,
                action_dim=action_dim,
                num_diffusion_iters=100,
                vision = True,
                cached_labels_path = '../output/cached_labels.pkl')
    policy.to(device)
    embedding_type = "learned" if concept_embeddings is not None else "original"
    #save episode scores to a file
    task_str = task[0] + str(task[1])
    #create the directory if it doesn't exist
    os.makedirs(os.path.join('../output/eval/', model_name,task_str,policy_type, init_type), exist_ok=True)
    f = open(os.path.join('../output/eval/', model_name,task_str,policy_type, init_type, 'episode_scores.txt'), 'w')
    f.write(f"Task: {task[0]} {corner_names[task[1]]}\n")
    f.write(f"Policy type: {policy_type}\n")
    f.write(f"Learning initial type: {init_type}\n")
    f.write(f"Concept weights: {concept_weights}\n")
    f.write(f"Episode scores: \n")
    # Load the saved weights
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    #test by replacing the concept weights with the embedding weights
    #check cached labels
    if embedding_type == "original":
        #test with one new concept
        # concept_embeddings = embedding_weights[1:2]
        description = f'push the {task[0]} block to the {corner_names[task[1]]} corner'
        print(description)
        concept_embeddings = policy.encode_text([description])
        concept_weights = torch.tensor([1.5], device=device)
    # environment setup
    tot_score = 0
    all_actions = []
    print(f"Using {embedding_type} concept weights")
    tot_steps = 0
    for episode in range(1,num_episodes+1):
        yaml_idx = task_names.index(task[0])
        cfg = OmegaConf.create(yamls[yaml_idx])
        env = ter_env.TEREnv(**cfg.info, scene_info=cfg.scene_info, agent_info=cfg.agent_info,
                             verbose = False)
        env.set_seed(np.random.randint(0,10000))
        env._setup()
        image_observer = ImageObserver(env, render_size=96, verbose=False)
        state_observer = StateObserver(env, verbose=False)
        def new_step(action):
            """Step the environment with the given action."""
            obs, reward, done, info = env.step(action)
            # get image observation
            img = image_observer.observe()
            state = state_observer.observe()    
            #done if in one corner
            if win_condition[task[1]](state):
                reward = 1
                print(f"Succeed!")
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
            # sample actions from the policy
            if concept_embeddings is not None and concept_weights is not None:
                # Use concept composition sampling
                c0_embedding = torch.zeros(concept_embeddings.shape[1])
                naction = sample_with_concept_composition(
                    policy=policy,
                    nimages=nimages,
                    nagent_poses=nagent_poses,
                    concept_embeddings=concept_embeddings,
                    concept_weights=concept_weights,
                    c0_embedding=c0_embedding,
                    num_diffusion_iters=100,
                    n_samples=10,
                    policy_type=policy_type,
                )   
            else:
                # Fall back to original sampling
                if policy_type == "diffusion":
                    naction = policy.sample(
                        nimages=nimages,
                        nagent_poses=nagent_poses,
                        num_diffusion_iters=100,
                        n_samples=1
                    )
                elif policy_type == "flow_matching":
                    naction = policy.sample(
                        nimages=nimages,
                        nagent_poses=nagent_poses,
                        nsamples=1
                    )
            if render and step_idx % 32 == 0:
                print(f"Saving episode {episode} gif")
                visualize_trajectories(
                    naction,
                    n=10,
                    gif_path=os.path.join(f'../output/eval/{model_name}', f'sample_trajectories.gif'),
                    fps=10,
                    seed=seed,
                    background_img = images[-1]
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
        print(f"Episode {episode} score: {sum(rewards)}, steps: {step_idx}")
        f.write(f"{sum(rewards)}\n")
        if render:
            print(f"Saving episode {episode} gif")
            images_to_gif(imgs, os.path.join('../output/eval/', model_name,task_str,policy_type, init_type,f'episode_{episode}.gif'), fps=10)
        if step_idx < max_steps:
            tot_steps += step_idx
    if render:
        #padding the last action to make all actions same length
        trajs = pad_and_stack_trajectories(all_actions)
        visualize_trajectories(
            trajs,
            n=num_episodes,
            gif_path=os.path.join('../output/eval/', model_name,task_str,policy_type, init_type, 'all_episodes.gif'),
            fps=10,
            seed=seed
        )
    print(f"Total score: {tot_score} over {num_episodes} episodes")
    print(f"Total steps: {tot_steps} over {tot_score} winning episodes")
    f.write(f"Total score: {tot_score} over {num_episodes} episodes\n")
    f.write(f"Total steps: {tot_steps} over {tot_score} winning episodes\n")
    f.close()
    return tot_score

def run_inference_flow():
    learned_weights = None
    embeddings = None
    evaluate(model_path='../output/blue_flow_policy_BERT.pth', 
                render=True, concept_embeddings=embeddings, concept_weights=learned_weights, policy_type="flow_matching", task = ['blue', 1])
    print("Inference completed.")
def run_inference_diffusion():
    # learned_weights = np.load("../output/concepts/blue_1_weights_diffusion.npy")
    # embeddings = np.load("../output/concepts/blue_1_embeddings_diffusion.npy")
    learned_weights = None
    embeddings = None
    evaluate(model_path='../output/blue_diffusion_policy_VLM2Vec.pth', 
                render=True, concept_embeddings=embeddings, concept_weights=learned_weights, policy_type="diffusion", task = ['blue', 0])
    print("Inference completed.")
if __name__ == "__main__":
    # evaluate the model
    # #test the model with the learned concept weights
    # embeddings = np.zeros((1,512))
    # learned_weights = np.array([1])
    run_inference_diffusion()
    # run_inference_flow()
    pass