#infering new concepts from a set of demonstrations using frozen diffusion policy or flow matching policy
import sys
import os
# Add parent directory to path so we can import diffusion_policy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from diffusion_policy.utils.img_to_gif import images_to_gif
import random
import collections
from tqdm import tqdm
from diffusion_policy.utils.visualization import visualize_trajectories, pad_and_stack_trajectories
from diffusion_policy.data.dataset import PushTImageDataset, normalize_data, unnormalize_data
from diffusion_policy.models.diffusion_policy import DiffusionPolicy
from diffusion_policy.models.flow_matching import FlowMatchingPolicy
import matplotlib.pyplot as plt
import umap
obs_horizon = 2  # number of observations to stack
pred_horizon = 16  # number of actions to predict
action_dim = 2  # action dimension, e.g. 2 for push task
action_horizon = 8  # number of actions to output, e.g. 8 for push task

dataset_path = '../output/save_data/test_workspace.pkl'
path1 = "../output/save_data/Blue_1.pkl"
path2 = "../output/save_data/Red_2.pkl"
path3 = "../output/save_data/Green_3.pkl"
dataset = PushTImageDataset([path1,path2,path3],['push the blue block to the upper-right corner',
    'push the red block to the upper-left corner',
    'push the green block to the lower-left corner'], 
                            pred_horizon=16, obs_horizon=2,action_horizon=8)
stats = dataset.stats
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_concept_loss_diffusion(diffusion_policy, nimage, nagent_pos, naction, concept_embeddings, concept_weights, 
                        c0_embedding, timesteps=None):
    """
    Compute the few-shot concept learning loss for diffusion policy from equation (2):
    E[||ε - (ε_θ(x_t(τ̃), c_0, s_0, t) + Σ(k=1 to K) ω_k(ε_θ(x_t(τ̃), c̃_k, s_0, t) - ε_θ(x_t(τ̃), c_0, s_0, t)))||²]
    
    Args:
        diffusion_policy: Frozen diffusion policy model
        nimage: shape (B, obs_horizon, C, H, W)
        nagent_pos: shape (B, obs_horizon, 2) 
        naction: shape (B, pred_horizon, action_dim)
        concept_embeddings: Fixed concept embeddings c̃_k, shape (K, embedding_dim)
        concept_weights: Learnable weights ω_k, shape (K,)
        c0_embedding: Base concept embedding c_0, shape (embedding_dim,)
        timesteps: Optional fixed timesteps
    """
    B = nimage.shape[0]
    K = len(concept_embeddings)
    
    # Sample noise
    noise = torch.randn_like(naction)
    
    # Sample random diffusion timesteps
    if timesteps is None:
        timesteps = torch.randint(
            0, diffusion_policy.noise_scheduler.config.num_train_timesteps,
            (B,), device=naction.device
        ).long()
    
    # Forward diffusion (q_sample)
    noisy_actions = diffusion_policy.noise_scheduler.add_noise(naction, noise, timesteps)
    
    # Encode observations 
    if diffusion_policy.vision:
        image_features = diffusion_policy.vision_encoder(nimage.flatten(end_dim=1))
        image_features = image_features.reshape(B, diffusion_policy.obs_horizon, -1)
        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
    else:
        obs_features = nagent_pos
    obs_cond_base = obs_features.flatten(start_dim=1)
    
    # Base prediction: ε_θ(x_t(τ̃), c_0, s_0, t)
    c0_emb_batch = c0_embedding.unsqueeze(0).repeat(B, 1)
    obs_cond_c0 = torch.cat([obs_cond_base, c0_emb_batch], dim=-1)
    eps_c0 = diffusion_policy.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond_c0)
    
    # Concept predictions: ε_θ(x_t(τ̃), c̃_k, s_0, t) for each concept k
    eps_concepts = []
    for k in range(K):
        ck_emb_batch = concept_embeddings[k].unsqueeze(0).repeat(B, 1)
        obs_cond_ck = torch.cat([obs_cond_base, ck_emb_batch], dim=-1)
        eps_ck = diffusion_policy.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond_ck)
        eps_concepts.append(eps_ck)
    
    # Compute weighted combination: ε_θ(x_t(τ̃), c_0, s_0, t) + Σ(k=1 to K) ω_k(ε_θ(x_t(τ̃), c̃_k, s_0, t) - ε_θ(x_t(τ̃), c_0, s_0, t))
    eps_pred = eps_c0.clone()
    for k in range(K):
        eps_pred += concept_weights[k] * (eps_concepts[k] - eps_c0)
    
    # Compute MSE loss
    loss = F.mse_loss(eps_pred, noise)
    
    return loss

def compute_concept_loss_flow_matching(flow_policy, nimage, nagent_pos, naction, concept_embeddings, concept_weights,
                                      c0_embedding, timesteps=None):
    """
    Compute the few-shot concept learning loss for flow matching policy:
    E[||v - (v_θ(x_t(τ̃), c_0, s_0, t) + Σ(k=1 to K) ω_k(v_θ(x_t(τ̃), c̃_k, s_0, t) - v_θ(x_t(τ̃), c_0, s_0, t)))||²]
    
    Where v is the true vector field and v_θ is the predicted vector field.
    
    Args:
        flow_policy: Frozen flow matching policy model
        nimage: shape (B, obs_horizon, C, H, W)
        nagent_pos: shape (B, obs_horizon, 2) 
        naction: shape (B, pred_horizon, action_dim)
        concept_embeddings: Fixed concept embeddings c̃_k, shape (K, embedding_dim)
        concept_weights: Learnable weights ω_k, shape (K,)
        c0_embedding: Base concept embedding c_0, shape (embedding_dim,)
        timesteps: Optional fixed timesteps (0-1 range for flow matching)
    """
    B = nimage.shape[0]
    K = len(concept_embeddings)
    
    # Sample random time points t ~ U(0, 1) for flow matching
    if timesteps is None:
        t = torch.rand(B, device=naction.device)  # (B,)
    else:
        t = timesteps
    
    # Sample noise (prior samples, typically Gaussian)  
    noise = torch.randn_like(naction)  # (B, pred_horizon, action_dim)
    
    # Create interpolated samples: x_t = (1-t) * noise + t * data
    x_t = (1 - t.unsqueeze(-1).unsqueeze(-1)) * noise + t.unsqueeze(-1).unsqueeze(-1) * naction
    
    # The true vector field for straight paths is: v_t = data - noise
    true_vector_field = naction - noise
    
    # Encode observations
    if flow_policy.vision:
        image_features = flow_policy.vision_encoder(nimage.flatten(end_dim=1))
        image_features = image_features.reshape(B, flow_policy.obs_horizon, -1)
        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
    else:
        obs_features = nagent_pos
    obs_cond_base = obs_features.flatten(start_dim=1)
    
    # Scale timesteps to 0-100 range (as expected by flow_predictor)
    timestep_scaled = t * 100
    
    # Base prediction: v_θ(x_t(τ̃), c_0, s_0, t)
    c0_emb_batch = c0_embedding.unsqueeze(0).repeat(B, 1)
    obs_cond_c0 = torch.cat([obs_cond_base, c0_emb_batch], dim=-1)
    v_c0 = flow_policy.flow_predictor(x_t, timestep=timestep_scaled, global_cond=obs_cond_c0)
    
    # Concept predictions: v_θ(x_t(τ̃), c̃_k, s_0, t) for each concept k
    v_concepts = []
    for k in range(K):
        ck_emb_batch = concept_embeddings[k].unsqueeze(0).repeat(B, 1)
        obs_cond_ck = torch.cat([obs_cond_base, ck_emb_batch], dim=-1)
        v_ck = flow_policy.flow_predictor(x_t, timestep=timestep_scaled, global_cond=obs_cond_ck)
        v_concepts.append(v_ck)
    
    # Compute weighted combination: v_θ(x_t(τ̃), c_0, s_0, t) + Σ(k=1 to K) ω_k(v_θ(x_t(τ̃), c̃_k, s_0, t) - v_θ(x_t(τ̃), c_0, s_0, t))
    v_pred = v_c0.clone()
    for k in range(K):
        v_pred += concept_weights[k] * (v_concepts[k] - v_c0)
    
    # Compute MSE loss between predicted and true vector field
    loss = F.mse_loss(v_pred, true_vector_field)
    
    return loss

def compute_concept_loss(policy, nimage, nagent_pos, naction, concept_embeddings, concept_weights, 
                        c0_embedding, policy_type="diffusion", timesteps=None):
    """
    Unified concept loss computation for both diffusion and flow matching policies.
    """
    if policy_type == "diffusion":
        return compute_concept_loss_diffusion(
            policy, nimage, nagent_pos, naction, concept_embeddings, concept_weights, c0_embedding, timesteps
        )
    elif policy_type == "flow_matching":
        return compute_concept_loss_flow_matching(
            policy, nimage, nagent_pos, naction, concept_embeddings, concept_weights, c0_embedding, timesteps
        )
    else:
        raise ValueError(f"Unsupported policy_type: {policy_type}. Must be 'diffusion' or 'flow_matching'")

# given a frozen diffusion policy or flow matching policy and a set of concepts, randomly initialize weights of concept, and
# gradient descent to optimize the weights of concept, and then use the optimized weights to generate new concepts
def infer_new_concepts(
    model_path: str = '../output/diffusion_policy.pth',
    policy_type: str = "diffusion",  # "diffusion" or "flow_matching"
    new_concept_dataset_path: str = '../output/save_data/new_concepts.pkl',
    weights_output_path: str = '../output/save_data/learned_concept_weights.npy',
    embeddings_output_path: str = '../output/save_data/learned_concept_embeddings.npy',
    num_epochs: int = 100,
    learning_rate: float = 0.002,
    batch_size: int = 512,
    K: int = 1,
    logging: bool = True):
    """
    Infer new concept weights from demonstrations using frozen policy (diffusion or flow matching).
    
    Args:
        model_path: Path to saved policy model
        policy_type: Type of policy - "diffusion" or "flow_matching"
        new_concept_dataset_path: Path to dataset with new concept demonstrations  
        num_epochs: Number of training epochs
        learning_rate: Learning rate for concept weight optimization
        batch_size: Batch size for training
        K: Number of concepts to learn
    """
    print(f"Loading {policy_type} policy...")
    
    # Load the appropriate policy type
    if policy_type == "diffusion":
        policy = DiffusionPolicy(
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            lowdim_obs_dim=2,
            action_dim=action_dim,
            num_diffusion_iters=100,
            vision=True,
            text=True
        )
    elif policy_type == "flow_matching":
        policy = FlowMatchingPolicy(
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            lowdim_obs_dim=2,
            action_dim=action_dim,
            num_diffusion_iters=100,
            vision=True,
            text=True
        )
    else:
        raise ValueError(f"Unsupported policy_type: {policy_type}")
    
    policy.to(device)
    
    # Load the saved weights
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    
    # Freeze the policy
    for param in policy.parameters():
        param.requires_grad = False
    policy.eval()
    
    print("Loading new concept dataset...")
    #load the new concept dataset
    new_dataset = PushTImageDataset([new_concept_dataset_path], [0], 
                                   pred_horizon=pred_horizon, obs_horizon=obs_horizon, action_horizon=action_horizon)
    # Load original semantic is CLIP embeddings of left/right concepts
    original_embeddings = policy.encode_text(['push the blue block to the upper-right corner',
    'push the red block to the upper-left corner',
    'push the green block to the lower-left corner'])
    print(f"Loaded original semantic embeddings shape: {original_embeddings.shape}")
    # randomly initialize the concept embeddings
    concept_embeddings = torch.randn(K, 512, device=device, requires_grad=True)
    #concept_embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device, requires_grad=True)
    
    # Initialize base concept c_0 as zero embeddings
    c0_embedding = torch.zeros(concept_embeddings.shape[1], device=device)
    
    print(f"Initializing {K} concept weights and making {K} concept embeddings trainable...")
    # set the concept weights as [1.5,1.5,1.5]
    concept_weights = torch.ones(K, device=device) * 1.5
    concept_weights.requires_grad = False
    # optimizer = torch.optim.Adam([concept_weights], lr=learning_rate)
    # Set up optimizer to optimize both concept weights and embeddings
    optimizer = torch.optim.Adam([concept_weights, concept_embeddings], lr=learning_rate)
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        new_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=8  
    )
    
    print(f"Starting concept weight optimization for {policy_type} policy...")
    # Progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    #use wandb to log the loss and weights
    if logging:
        import wandb
        wandb.init(project="concept-learning", name=f"concept-learning-{policy_type}")
        wandb.config.update({
        "num_epochs": num_epochs, 
        "learning_rate": learning_rate, 
        "batch_size": batch_size,
        "policy_type": policy_type
    })
    
    # Store trajectory of concept embeddings for UMAP visualization
    embedding_trajectory = []
    weight_trajectory = []
    for epoch in epoch_pbar:
        epoch_loss = 0.0
        num_batches = 0
        
        # Progress bar for batches within each epoch
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, unit="batch")
        
        for batch_idx, batch in enumerate(batch_pbar):
            optimizer.zero_grad()
            
            # Extract batch data
            nimage = batch['image'].to(device)  # (B, obs_horizon, C, H, W)
            nagent_pos = batch['agent_pos'].to(device)  # (B, obs_horizon, 2)
            naction = batch['action'].to(device)  # (B, pred_horizon, action_dim)
            
            # Compute concept learning loss
            loss = compute_concept_loss(
                policy=policy,
                nimage=nimage,
                nagent_pos=nagent_pos, 
                naction=naction,
                concept_embeddings=concept_embeddings,
                concept_weights=concept_weights,
                c0_embedding=c0_embedding,
                policy_type=policy_type
            )
            if logging:
                wandb.log({"loss": loss.item(), "epoch": epoch})
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update batch progress bar with current loss
            batch_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        # Update epoch progress bar with average loss and current weights
        # Compute embedding norms to track how much concepts are changing
        embedding_norms = torch.norm(concept_embeddings, dim=1).detach().cpu().numpy()
        
        if logging:
            wandb.log({
                "avg_loss": avg_loss,
                "weights": concept_weights.detach().cpu().numpy(),
            })
        
        # Store current embeddings and weights for trajectory visualization
        embedding_trajectory.append(concept_embeddings.detach().cpu().numpy().copy())
        weight_trajectory.append(concept_weights.detach().cpu().numpy().copy())
        
        concept_weights_str = ', '.join([f'{w:.3f}' for w in concept_weights.detach().cpu().numpy()])
        embedding_norms_str = ', '.join([f'{n:.3f}' for n in embedding_norms])
        
        epoch_pbar.set_postfix({
            'Avg Loss': f'{avg_loss:.6f}',
            'Weights': f'[{concept_weights_str}]',
            'Emb_Norms': f'[{embedding_norms_str}]'
        })
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch+1}: Weights=[{concept_weights_str}], Embedding_Norms=[{embedding_norms_str}]")
            
            # Create UMAP visualization of concept embedding trajectory
            if len(embedding_trajectory) > 1:
                # Combine original embeddings and learning trajectory for UMAP fitting
                all_embeddings = []
                
                # Add original semantic embeddings (left/right concepts)
                all_embeddings.extend(original_embeddings)
                
                # Add all embeddings from the learning trajectory 
                for step_embs in embedding_trajectory:
                    all_embeddings.extend(step_embs)
                
                all_embeddings = np.array([emb.cpu().numpy() if torch.is_tensor(emb) else emb for emb in all_embeddings])
                
                # Fit UMAP on combined data
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.1)
                embedding_2d = reducer.fit_transform(all_embeddings)
                
                # Split back into original and trajectory parts
                n_original = len(original_embeddings)
                n_concepts = len(concept_embeddings)
                n_steps = len(embedding_trajectory)
                
                original_2d = embedding_2d[:n_original]
                trajectory_2d = embedding_2d[n_original:].reshape(n_steps, n_concepts, 2)
                
                # Create visualization
                plt.figure(figsize=(10, 8))
                
                # Plot original semantic concepts (left/right)
                colors_orig = ['blue', 'green']
                labels_orig = ['Left Concept', 'Right Concept'] 
                for i, (pos, color, label) in enumerate(zip(original_2d, colors_orig, labels_orig)):
                    plt.scatter(pos[0], pos[1], c=color, s=500, marker='*', 
                              label=label, edgecolors='black', linewidth=2, alpha=0.8)
                
                # Plot trajectory for each learned concept
                concept_colors = ['red', 'orange']
                for c in range(n_concepts):
                    traj_c = trajectory_2d[:, c, :]  # trajectory for concept c
                    
                    # Plot trajectory path
                    plt.plot(traj_c[:, 0], traj_c[:, 1], 
                            color=concept_colors[c], alpha=0.6, linewidth=2,
                            label=f'Learned Concept {c+1} Trajectory')
                    
                    # Plot current position
                    plt.scatter(traj_c[-1, 0], traj_c[-1, 1], 
                              c=concept_colors[c], s=100, marker='o',
                              edgecolors='black', linewidth=1)
                    
                    # Plot starting position  
                    plt.scatter(traj_c[0, 0], traj_c[0, 1],
                              c=concept_colors[c], s=80, marker='s', alpha=0.5,
                              edgecolors='black', linewidth=1)
                
                plt.title(f"UMAP: Concept Embedding Evolution ({policy_type.title()}, Epoch {epoch+1})")
                plt.xlabel("UMAP Dimension 1")
                plt.ylabel("UMAP Dimension 2") 
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save to wandb
                if logging:
                    wandb.log({"concept_embedding_umap": wandb.Image(plt)})
                plt.close()
            
            # Also create weight trajectory plot
            if len(weight_trajectory) > 1:
                weight_traj_arr = np.array(weight_trajectory)
                plt.figure(figsize=(8, 6))
                
                if weight_traj_arr.shape[1] == 1:
                    # Single concept case - plot weight vs epoch
                    epochs = np.arange(len(weight_traj_arr))
                    plt.plot(epochs, weight_traj_arr[:, 0], 
                            marker='o', markersize=3, linewidth=1, alpha=0.7)
                    plt.scatter(epochs[-1], weight_traj_arr[-1, 0], 
                              c='red', s=50, label='Current Weight')
                    plt.scatter(epochs[0], weight_traj_arr[0, 0],
                              c='green', s=50, label='Initial Weight')
                    plt.title(f"Weight Convergence Over Time ({policy_type.title()}, Epoch {epoch+1})")
                    plt.xlabel("Training Step")
                    plt.ylabel("Weight Value")
                elif weight_traj_arr.shape[1] >= 2:
                    # Multiple concepts case - plot in weight space
                    plt.plot(weight_traj_arr[:, 0], weight_traj_arr[:, 1], 
                            marker='o', markersize=3, linewidth=1, alpha=0.7)
                    plt.scatter(weight_traj_arr[-1, 0], weight_traj_arr[-1, 1], 
                              c='red', s=50, label='Current Weights')
                    plt.scatter(weight_traj_arr[0, 0], weight_traj_arr[0, 1],
                              c='green', s=50, label='Initial Weights')
                    plt.title(f"Weight Space Convergence ({policy_type.title()}, Epoch {epoch+1})")
                    plt.xlabel("Weight 1")
                    plt.ylabel("Weight 2")
                
                plt.grid(True, alpha=0.3)
                plt.legend()
                if logging:
                    wandb.log({"weight_trajectory": wandb.Image(plt)})
                plt.close()
            
            # Save current state
            np.save(weights_output_path, concept_weights.detach().cpu().numpy())
            np.save(embeddings_output_path, concept_embeddings.detach().cpu().numpy())
    
    print("Concept weight optimization completed!")
    print(f"Final concept weights: {concept_weights.detach().cpu().numpy()}")
    
    np.save(weights_output_path, concept_weights.detach().cpu().numpy())
    np.save(embeddings_output_path, concept_embeddings.detach().cpu().numpy())
    
    print(f"Learned concept weights saved to: {weights_output_path}")
    print(f"Learned concept embeddings saved to: {embeddings_output_path}")
    
    return {
        'weights': concept_weights.detach().cpu().numpy(),
        'embeddings': concept_embeddings.detach().cpu().numpy()
    }

if __name__ == "__main__":
    # Example usage for diffusion policy
    print("Running concept inference for diffusion policy...")
    learned_weights_diffusion = infer_new_concepts(
        model_path='../output/diffusion_policy_push.pth',
        policy_type="diffusion",
        new_concept_dataset_path='../output/save_data/Blue_1.pkl',
        weights_output_path='../output/concepts/blue_weights_diffusion.npy',
        embeddings_output_path='../output/concepts/blue_embeddings_diffusion.npy'
    )
    print("Diffusion policy concept learning completed. Learned weights:", learned_weights_diffusion)
    
    # Example usage for flow matching policy  
    # print("\nRunning concept inference for flow matching policy...")
    # learned_weights_flow = infer_new_concepts(
    #     model_path='../output/flow_policy.pth',
    #     policy_type="flow_matching",
    #     new_concept_dataset_path='../output/save_data/left.pkl',
    #     weights_output_path='../output/concepts/left_weights_flow.npy',
    #     embeddings_output_path='../output/concepts/left_embeddings_flow.npy',
    #     logging = True
    # )
    # print("Flow matching policy concept learning completed. Learned weights:", learned_weights_flow)

