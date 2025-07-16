#infering new concepts from a set of demonstrations using frozen diffusion policy or flow matching policy
# 
# This script includes UMAP visualization functionality that shows:
# 1. The trajectory of learned concept embeddings during training
# 2. All 12 original embeddings (3 colors × 4 corners) as reference points  
# 3. Weight evolution plots showing how concept weights change over epochs
#
# The UMAP plots will automatically be saved alongside the learned weights and embeddings.
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

# Constants
CORNER_NAMES = ['lower-right', 'upper-right', 'upper-left', 'lower-left']
COLORS = ['Blue', 'Red', 'Green']
COLORS_LOWER = ['blue', 'red', 'green']
OBS_HORIZON = 2  # number of observations to stack
PRED_HORIZON = 16  # number of actions to predict
ACTION_DIM = 2  # action dimension, e.g. 2 for push task
ACTION_HORIZON = 8  # number of actions to output, e.g. 8 for push task
EMBEDDING_DIM = 64

# Paths
DATASET_PATH = '../output/save_data/test_workspace.pkl'
CONCEPTS_DIR = '../output/concepts'
PATH_LIST = [f'../dataset/{color}_{num}.pkl' for color in COLORS for num in [0,1,2,3]]
DESCRIPTION_LIST = [f'push the {color} block to the {corner} corner' for color in COLORS_LOWER for corner in CORNER_NAMES]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_obs_cond_and_prediction(policy, nimage, nagent_pos, embeddings, uncond=False):
    """Helper function to get observation conditioning and model prediction."""
    obs_cond = policy.get_cond(nimage, nagent_pos, embeddings, uncond=uncond)
    return obs_cond

def compute_concept_loss_diffusion(diffusion_policy, nimage, nagent_pos, naction, concept_embeddings, concept_weights, 
                        c0_embedding, timesteps=None):
    """
    Compute the few-shot concept learning loss for diffusion policy from equation (2):
    E[||ε - (ε_θ(x_t(τ̃), c_0, s_0, t) + Σ(k=1 to K) ω_k(ε_θ(x_t(τ̃), c̃_k, s_0, t) - ε_θ(x_t(τ̃), c_0, s_0, t)))||²]
    """
    B = nimage.shape[0]
    K = len(concept_embeddings)
    
    # Sample noise and timesteps
    noise = torch.randn_like(naction)
    if timesteps is None:
        timesteps = torch.randint(
            0, diffusion_policy.noise_scheduler.config.num_train_timesteps,
            (B,), device=naction.device
        ).long()
    
    # Forward diffusion (q_sample)
    noisy_actions = diffusion_policy.noise_scheduler.add_noise(naction, noise, timesteps)
    
    # Get base prediction without text conditioning
    obs_cond_c0 = get_obs_cond_and_prediction(diffusion_policy, nimage, nagent_pos, None, uncond=True)
    eps_c0 = diffusion_policy.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond_c0)
    
    # Compute weighted combination of concept predictions
    eps_pred = eps_c0.clone()
    for k in range(K):
        batch_concept_embeddings = concept_embeddings[k].unsqueeze(0).repeat(B, 1)
        obs_cond_ck = get_obs_cond_and_prediction(diffusion_policy, nimage, nagent_pos, batch_concept_embeddings, uncond=False)
        eps_ck = diffusion_policy.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond_ck)
        eps_pred += concept_weights[k] * (eps_ck - eps_c0)
    
    return F.mse_loss(eps_pred, noise)

def compute_concept_loss_flow_matching(flow_policy, nimage, nagent_pos, naction, concept_embeddings, concept_weights,
                                      c0_embedding, timesteps=None):
    """
    Compute the few-shot concept learning loss for flow matching policy:
    E[||v - (v_θ(x_t(τ̃), c_0, s_0, t) + Σ(k=1 to K) ω_k(v_θ(x_t(τ̃), c̃_k, s_0, t) - v_θ(x_t(τ̃), c_0, s_0, t)))||²]
    """
    B = nimage.shape[0]
    K = len(concept_embeddings)
    
    # Sample random time points and noise for flow matching
    t = torch.rand(B, device=naction.device) if timesteps is None else timesteps
    noise = torch.randn_like(naction)
    
    # Create interpolated samples and true vector field
    x_t = (1 - t.unsqueeze(-1).unsqueeze(-1)) * noise + t.unsqueeze(-1).unsqueeze(-1) * naction
    true_vector_field = naction - noise
    
    # Get base prediction without text conditioning
    obs_cond_c0 = get_obs_cond_and_prediction(flow_policy, nimage, nagent_pos, None, uncond=True)
    v_c0 = flow_policy.flow_predictor(x_t, timestep=t*100, global_cond=obs_cond_c0)
    
    # Compute weighted combination of concept predictions
    v_pred = v_c0.clone()
    for k in range(K):
        batch_concept_embeddings = concept_embeddings[k].unsqueeze(0).repeat(B, 1)
        obs_cond_ck = get_obs_cond_and_prediction(flow_policy, nimage, nagent_pos, batch_concept_embeddings, uncond=False)
        v_ck = flow_policy.flow_predictor(x_t, timestep=t*100, global_cond=obs_cond_ck)
        v_pred += concept_weights[k] * (v_ck - v_c0)
    
    return F.mse_loss(v_pred, true_vector_field)

def compute_concept_loss(policy, nimage, nagent_pos, naction, concept_embeddings, concept_weights, 
                        c0_embedding, policy_type="diffusion", timesteps=None):
    """Unified concept loss computation for both diffusion and flow matching policies."""
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

def initialize_concept_embeddings(policy, init_type, K, task, description_list):
    """Initialize concept embeddings based on the specified type."""
    if init_type == "rand":
        return torch.randn(K, EMBEDDING_DIM, device=device, requires_grad=True)
    elif init_type == 'average':
        weights = torch.randn(K, len(description_list), device=device)
        weights = F.softmax(weights, dim=1)  # Use softmax for proper normalization
        original_embeddings = policy.encode_text(description_list)
        concept_embeddings = weights @ original_embeddings
        return concept_embeddings.detach().clone().requires_grad_(True)
    elif init_type == 'zero':
        return torch.zeros(K, EMBEDDING_DIM, device=device, requires_grad=True)
    elif init_type == 'simplex':
        # For simplex, we return the original embeddings and will optimize combination weights separately
        original_embeddings = policy.encode_text(description_list)
        return original_embeddings.detach()  # These won't be optimized directly
    else:  # init_type in ['fixed', 'task']
        concept_embeddings = policy.encode_text([f'push the {task[0]} block to the {CORNER_NAMES[task[1]]} corner'])
        concept_embeddings = concept_embeddings.detach().clone().requires_grad_(init_type == 'fixed')
        return concept_embeddings

def setup_logging(logging, policy_type, init_type, task, num_epochs, learning_rate, batch_size):
    """Setup wandb logging if enabled."""
    if logging:
        import wandb
        wandb.init(project="concept-learning", name=f"{policy_type}-{init_type}-{task[0]}-{task[1]}")
        wandb.config.update({
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "policy_type": policy_type
        })

def log_batch_progress(logging, loss, epoch, scheduler):
    """Log batch-level training progress to wandb if enabled."""
    if logging:
        import wandb
        wandb.log({
            "loss": loss.item(),
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0]
        })

def log_epoch_progress(logging, epoch, scheduler, avg_loss, concept_weights):
    """Log epoch-level training progress to wandb if enabled."""
    if logging:
        import wandb
        wandb.log({
            "avg_loss": avg_loss,
            "weights": concept_weights.detach().cpu().numpy(),
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0]
        })

def save_visualizations(embedding_trajectory, weight_trajectory, original_embeddings, weights_output_path, 
                       policy_type, init_type, logging):
    """Generate and save UMAP and weight trajectory visualizations."""
    umap_save_path = weights_output_path.replace('_weights.npy', '_umap_trajectory.png')
    weight_plot_save_path = weights_output_path.replace('_weights.npy', '_weight_trajectory.png')
    
    try:
        visualize_embedding_trajectory_umap(
            embedding_trajectory=embedding_trajectory,
            weight_trajectory=weight_trajectory, 
            original_embeddings=original_embeddings,
            description_list=[f'{color}_{num}' for color in COLORS_LOWER for num in range(4)],
            save_path=umap_save_path,
            title=f"Embedding Trajectory - {policy_type.title()} Policy ({init_type} init)"
        )
        
        plot_weight_trajectory(
            weight_trajectory=weight_trajectory,
            save_path=weight_plot_save_path,
            title=f"Weight Evolution - {policy_type.title()} Policy ({init_type} init)"
        )
        
        if logging:
            import wandb
            wandb.log({
                "embedding_trajectory": wandb.Image(umap_save_path), 
                "weight_trajectory": wandb.Image(weight_plot_save_path)
            })
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")

def visualize_embedding_trajectory_umap(embedding_trajectory, weight_trajectory, original_embeddings, 
                                       description_list, save_path=None, title="Embedding Trajectory"):
    """
    Visualize the trajectory of concept embeddings using UMAP, with original embeddings as reference points.
    """
    # Convert inputs to numpy
    if torch.is_tensor(original_embeddings):
        original_embeddings = original_embeddings.detach().cpu().numpy()
    
    # Flatten all trajectory embeddings
    all_trajectory_embeddings = []
    trajectory_epoch_labels = []
    trajectory_concept_labels = []
    
    for epoch, embeddings in enumerate(embedding_trajectory):
        for concept_idx in range(embeddings.shape[0]):
            all_trajectory_embeddings.append(embeddings[concept_idx])
            trajectory_epoch_labels.append(epoch)
            trajectory_concept_labels.append(concept_idx)
    
    all_trajectory_embeddings = np.array(all_trajectory_embeddings)
    
    # Combine all embeddings for UMAP fitting
    all_embeddings = np.vstack([original_embeddings, all_trajectory_embeddings])
    
    # Fit UMAP
    print("Fitting UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(all_embeddings)
    
    # Split back into original and trajectory embeddings
    original_2d = embedding_2d[:len(original_embeddings)]
    trajectory_2d = embedding_2d[len(original_embeddings):]
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot original embeddings as reference points
    scatter_orig = plt.scatter(original_2d[:, 0], original_2d[:, 1], 
                              c='black', s=100, marker='*', alpha=0.8, 
                              label='Original Embeddings', zorder=3)
    
    # Add labels for original embeddings
    for i, desc in enumerate(description_list):
        # Truncate description for readability
        short_desc = desc.replace('push the ', '').replace(' block to the ', ' → ').replace(' corner', '')
        plt.annotate(short_desc, (original_2d[i, 0], original_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    # Plot trajectory for each concept
    K = embedding_trajectory[0].shape[0] if embedding_trajectory else 1
    colors_trajectory = plt.cm.tab10(np.linspace(0, 1, K))
    
    for concept_idx in range(K):
        # Get indices for this concept
        concept_mask = np.array(trajectory_concept_labels) == concept_idx
        concept_trajectory_2d = trajectory_2d[concept_mask]
        concept_epochs = np.array(trajectory_epoch_labels)[concept_mask]
        
        if len(concept_trajectory_2d) > 0:
            # Plot trajectory line
            plt.plot(concept_trajectory_2d[:, 0], concept_trajectory_2d[:, 1], 
                    color=colors_trajectory[concept_idx], alpha=0.6, linewidth=2,
                    label=f'Concept {concept_idx} Trajectory')
            
            # Plot start point
            plt.scatter(concept_trajectory_2d[0, 0], concept_trajectory_2d[0, 1],
                       color=colors_trajectory[concept_idx], s=150, marker='o', 
                       edgecolor='white', linewidth=2, zorder=4,
                       label=f'Concept {concept_idx} Start' if concept_idx == 0 else "")
            
            # Plot end point
            plt.scatter(concept_trajectory_2d[-1, 0], concept_trajectory_2d[-1, 1],
                       color=colors_trajectory[concept_idx], s=150, marker='s', 
                       edgecolor='white', linewidth=2, zorder=4,
                       label=f'Concept {concept_idx} End' if concept_idx == 0 else "")
            
            # Add epoch annotations at regular intervals
            step = max(1, len(concept_trajectory_2d) // 5)  # Show ~5 annotations
            for i in range(0, len(concept_trajectory_2d), step):
                plt.annotate(f'{concept_epochs[i]}', 
                           (concept_trajectory_2d[i, 0], concept_trajectory_2d[i, 1]),
                           xytext=(3, 3), textcoords='offset points', 
                           fontsize=6, alpha=0.5, color=colors_trajectory[concept_idx])
    
    plt.title(title, fontsize=14)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"UMAP visualization saved to: {save_path}")
    
    plt.show()

def plot_weight_trajectory(weight_trajectory, save_path=None, title="Weight Trajectory"):
    """Plot the evolution of concept weights over epochs."""
    weight_trajectory = np.array(weight_trajectory)  # Shape: (num_epochs, K)
    epochs = np.arange(len(weight_trajectory))
    K = weight_trajectory.shape[1]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    
    for k in range(K):
        plt.plot(epochs, weight_trajectory[:, k], color=colors[k], 
                linewidth=2, label=f'Concept {k} Weight')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Weight trajectory plot saved to: {save_path}")
    
    plt.show()

def infer_new_concepts(
    model_path: str = '../output/diffusion_policy.pth',
    policy_type: str = "diffusion",  # "diffusion" or "flow_matching"
    new_concept_dataset_path: str = '../output/save_data/new_concepts.pkl',
    weights_output_path: str = '../output/save_data/learned_concept_weights.npy',
    embeddings_output_path: str = '../output/save_data/learned_concept_embeddings.npy',
    num_epochs: int = 100,
    learning_rate: float = 1e-2,
    batch_size: int = 256,
    K: int = 1,
    logging: bool = True,
    init_type: str = "rand",
    task: list = ['blue', 1],
    noise_pred_net_type: str = 'unet',
    num_trajectories: int = None):
    """
    Infer new concept weights from demonstrations using frozen policy (diffusion or flow matching).
    """
    print(f"Loading {policy_type} policy...")
    
    # Load the appropriate policy type
    if policy_type == "diffusion":
        policy = DiffusionPolicy(
            obs_horizon=OBS_HORIZON,
            pred_horizon=PRED_HORIZON,
            lowdim_obs_dim=2,
            action_dim=ACTION_DIM,
            num_diffusion_iters=100,
            vision=True,
            text=True,
            cached_labels_path='../output/cached_labels.pkl',
            noise_pred_net_type=noise_pred_net_type
        )
    elif policy_type == "flow_matching":
        policy = FlowMatchingPolicy(
            obs_horizon=OBS_HORIZON,
            pred_horizon=PRED_HORIZON,
            lowdim_obs_dim=2,
            action_dim=ACTION_DIM,
            num_diffusion_iters=100,
            vision=True,
            text=True,
            cached_labels_path='../output/cached_labels.pkl'
        )
    else:
        raise ValueError(f"Unsupported policy_type: {policy_type}")
    
    policy.to(device)
    
    # Load the saved weights and freeze the policy
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    for param in policy.parameters():
        param.requires_grad = False
    policy.eval()
    
    print("Loading new concept dataset...")
    new_dataset = PushTImageDataset([new_concept_dataset_path], [0], 
                                   pred_horizon=PRED_HORIZON, obs_horizon=OBS_HORIZON, 
                                   action_horizon=ACTION_HORIZON, num_trajectories=num_trajectories)
    
    # Initialize concept embeddings and weights
    print(f"Initializing {init_type} concept embeddings")
    original_embeddings = policy.encode_text(DESCRIPTION_LIST).detach()  # Detach to avoid grad issues
    concept_embeddings = initialize_concept_embeddings(policy, init_type, K, task, DESCRIPTION_LIST)
    
    if init_type in ['fixed', 'task'] and concept_embeddings.shape[0] == 1:
        K = 1
    
    # Handle simplex initialization
    if init_type == 'simplex':
        # Initialize simplex weights for linear combination of original embeddings
        num_original = len(DESCRIPTION_LIST)  # 12 original embeddings
        simplex_weights = torch.randn(K, num_original, device=device, requires_grad=True)
        # Normalize to create proper simplex (weights sum to 1)
        simplex_weights.data = F.softmax(simplex_weights.data, dim=1)
        print(f"Initialized {K} simplex weight vectors of size {num_original}")
        print(f"Initial simplex weights shape: {simplex_weights.shape}")
        optimizer_params = [simplex_weights]
    else:
        simplex_weights = None
        optimizer_params = [concept_embeddings]
    
    c0_embedding = torch.zeros(concept_embeddings.shape[1], device=device)
    concept_weights = torch.ones(K, device=device) * 1.5 / K
    concept_weights.requires_grad = False
    
    print(f"Initializing {K} concept weights: {concept_weights}")
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.Adam(optimizer_params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        new_dataset, batch_size=batch_size, shuffle=True, num_workers=8  
    )
    
    # Setup logging
    setup_logging(logging, policy_type, init_type, task, num_epochs, learning_rate, batch_size)
    
    print(f"Starting concept weight optimization for {policy_type} policy...")
    
    # Training loop
    embedding_trajectory = []
    weight_trajectory = []
    
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    
    for epoch in epoch_pbar:
        epoch_loss = 0.0
        num_batches = 0
        
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, unit="batch")
        
        for batch_idx, batch in enumerate(batch_pbar):
            optimizer.zero_grad()
            
            # For simplex initialization, compute concept embeddings as linear combinations
            if init_type == 'simplex':
                # Normalize simplex weights to ensure they sum to 1
                normalized_simplex_weights = F.softmax(simplex_weights, dim=1)
                # Compute concept embeddings as weighted combinations of original embeddings  
                current_concept_embeddings = normalized_simplex_weights @ original_embeddings
            else:
                current_concept_embeddings = concept_embeddings
            
            # Extract batch data
            nimage = batch['image'].to(device)
            nagent_pos = batch['agent_pos'].to(device)
            naction = batch['action'].to(device)
            
            # Compute concept learning loss
            loss = compute_concept_loss(
                policy=policy, nimage=nimage, nagent_pos=nagent_pos, naction=naction,
                concept_embeddings=current_concept_embeddings, concept_weights=concept_weights,
                c0_embedding=c0_embedding, policy_type=policy_type
            )
            
            log_batch_progress(logging, loss, epoch, scheduler)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            batch_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
         
         # Compute current concept embeddings for trajectory and progress tracking
        if init_type == 'simplex':
            normalized_simplex_weights = F.softmax(simplex_weights, dim=1)
            current_concept_embeddings_for_tracking = normalized_simplex_weights @ original_embeddings
        else:
            current_concept_embeddings_for_tracking = concept_embeddings
         
         # Store trajectory and update progress
        embedding_trajectory.append(current_concept_embeddings_for_tracking.detach().cpu().numpy().copy())
        weight_trajectory.append(concept_weights.detach().cpu().numpy().copy())
        
        embedding_norms = torch.norm(current_concept_embeddings_for_tracking, dim=1).detach().cpu().numpy()
        concept_weights_str = ', '.join([f'{w:.3f}' for w in concept_weights.detach().cpu().numpy()])
        embedding_norms_str = ', '.join([f'{n:.3f}' for n in embedding_norms])
        
        log_epoch_progress(logging, epoch, scheduler, avg_loss, concept_weights)
        
        epoch_pbar.set_postfix({
            'Avg Loss': f'{avg_loss:.6f}',
            'Weights': f'[{concept_weights_str}]',
            'Emb_Norms': f'[{embedding_norms_str}]'
        })
        
        if epoch == num_epochs - 1:
            print(f"Final - Weights=[{concept_weights_str}], Embedding_Norms=[{embedding_norms_str}]")
    
    print("Concept weight optimization completed!")
    
    # Compute final concept embeddings for simplex case
    if init_type == 'simplex':
        normalized_simplex_weights = F.softmax(simplex_weights, dim=1)
        final_concept_embeddings = normalized_simplex_weights @ original_embeddings
        print(f"Final simplex weights shape: {normalized_simplex_weights.shape}")
        print(f"Final simplex weights (first concept): {normalized_simplex_weights[0].detach().cpu().numpy()}")
    else:
        final_concept_embeddings = concept_embeddings
    
    # Save results
    os.makedirs(os.path.dirname(weights_output_path), exist_ok=True)
    np.save(weights_output_path, concept_weights.detach().cpu().numpy())
    np.save(embeddings_output_path, final_concept_embeddings.detach().cpu().numpy())
    
    # For simplex, also save the combination weights
    if init_type == 'simplex':
        simplex_weights_path = weights_output_path.replace('_weights.npy', '_simplex_weights.npy')
        np.save(simplex_weights_path, normalized_simplex_weights.detach().cpu().numpy())
        print(f"Simplex combination weights saved to: {simplex_weights_path}")
    
    print(f"Results saved to: {weights_output_path}, {embeddings_output_path}")
    
    # Generate visualizations
    print("Generating visualizations...")
    save_visualizations(embedding_trajectory, weight_trajectory, original_embeddings, 
                       weights_output_path, policy_type, init_type, logging)
    
    if logging:
        import wandb
        wandb.finish()
    
    return {
        'weights': concept_weights.detach().cpu().numpy(),
        'embeddings': final_concept_embeddings.detach().cpu().numpy(),
        'embedding_trajectory': embedding_trajectory,
        'weight_trajectory': weight_trajectory,
        'simplex_weights': normalized_simplex_weights.detach().cpu().numpy() if init_type == 'simplex' else None
    }

def run_concept_inference(policy_type="diffusion", task_name="blue_2", **kwargs):
    """
    Unified function to run concept inference for both diffusion and flow matching policies.
    
    Args:
        policy_type: "diffusion" or "flow_matching"
        task_name: Name identifier for the task (used in file naming)
        **kwargs: Additional arguments passed to infer_new_concepts
    """
    print(f"Running concept inference for {policy_type} policy...")
    
    # Ensure concepts directory exists
    os.makedirs(CONCEPTS_DIR, exist_ok=True)
    
    # Set default parameters based on policy type
    default_params = {
        'model_path': f'../output/{policy_type}_policy_push.pth',
        'policy_type': policy_type,
        'new_concept_dataset_path': '../output/save_data/Blue_2.pkl',
        'weights_output_path': f'{CONCEPTS_DIR}/{task_name}_weights_{policy_type}.npy',
        'embeddings_output_path': f'{CONCEPTS_DIR}/{task_name}_embeddings_{policy_type}.npy',
        'init_type': "rand",
        'num_epochs': 500,
        'K': 2,
    }
    
    # Override with any provided kwargs
    default_params.update(kwargs)
    
    learned_weights = infer_new_concepts(**default_params)
    print(f"{policy_type.title()} policy concept learning completed.")
    return learned_weights

if __name__ == "__main__":
    # Example usage - can run either policy type with the unified function
    # run_concept_inference(policy_type="diffusion", task_name="blue_2")
    # run_concept_inference(policy_type="flow_matching", task_name="blue_2")
    
    # Test the new simplex initialization type
    run_concept_inference(
        policy_type="flow_matching", 
        task_name="blue_2_simplex",
        init_type="simplex",
        num_epochs=500,
        K=2
    )

