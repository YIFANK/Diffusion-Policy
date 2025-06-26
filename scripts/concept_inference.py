#infering new concepts from a set of demonstrations using frozen diffusion policy
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

obs_horizon = 2  # number of observations to stack
pred_horizon = 16  # number of actions to predict
action_dim = 2  # action dimension, e.g. 2 for push task
action_horizon = 8  # number of actions to output, e.g. 8 for push task

dataset_path = '../output/save_data/test_workspace.pkl'
path1 = "../output/save_data/left.pkl"
path2 = "../output/save_data/right.pkl"
dataset = PushTImageDataset([path1,path2],[-1,1], 
                            pred_horizon=16, obs_horizon=2,action_horizon=8)
stats = dataset.stats
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embeddings = np.load("../output/save_data/embeddings.npy")

def compute_concept_loss(diffusion_policy, nimage, nagent_pos, naction, concept_embeddings, concept_weights, 
                        c0_embedding, timesteps=None):
    """
    Compute the few-shot concept learning loss from equation (2):
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

# given a frozen diffusion policy and a set of concepts, randomly initialize weights of concept, and
# gradient descent to optimize the weights of concept, and then use the optimized weights to generate new concepts
def infer_new_concepts(
    model_path: str = '../output/diffusion_policy.pth',
    new_concept_dataset_path: str = '../output/save_data/new_concepts.pkl',
    num_epochs: int = 500,
    learning_rate: float = 0.001,
    batch_size: int = 32):
    """
    Infer new concept weights from demonstrations using frozen diffusion policy.
    
    Args:
        model_path: Path to saved diffusion policy model
        new_concept_dataset_path: Path to dataset with new concept demonstrations  
        num_epochs: Number of training epochs
        learning_rate: Learning rate for concept weight optimization
        batch_size: Batch size for training
    """
    print("Loading diffusion policy...")
    #load the diffusion policy
    diffusion_policy = DiffusionPolicy(
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        lowdim_obs_dim=2,
        action_dim=action_dim,
        num_diffusion_iters=100,
        vision=False,
        text=True
    )
    diffusion_policy.to(device)
    
    # Load the saved weights
    state_dict = torch.load(model_path, map_location=device)
    diffusion_policy.load_state_dict(state_dict)
    
    # Freeze the diffusion policy
    for param in diffusion_policy.parameters():
        param.requires_grad = False
    diffusion_policy.eval()
    
    print("Loading new concept dataset...")
    #load the new concept dataset
    new_dataset = PushTImageDataset([new_concept_dataset_path], [0], 
                                   pred_horizon=pred_horizon, obs_horizon=obs_horizon, action_horizon=action_horizon)
    # Convert embeddings to torch tensors
    concept_embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
    K = len(concept_embeddings)  # Number of concepts
    
    # Initialize base concept c_0 as zero embedding or mean of concepts
    c0_embedding = torch.zeros(concept_embeddings.shape[1], device=device)
    # Alternative: c0_embedding = concept_embeddings.mean(dim=0)
    
    print(f"Initializing {K} concept weights...")
    # Randomly initialize the concept weights ω_k
    concept_weights = torch.randn(K, device=device, requires_grad=True)
    
    # Set up optimizer to only optimize concept weights
    optimizer = torch.optim.Adam([concept_weights], lr=learning_rate)
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        new_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    print("Starting concept weight optimization...")
    # Progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    
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
                diffusion_policy=diffusion_policy,
                nimage=nimage,
                nagent_pos=nagent_pos, 
                naction=naction,
                concept_embeddings=concept_embeddings,
                concept_weights=concept_weights,
                c0_embedding=c0_embedding
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update batch progress bar with current loss
            batch_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        # Update epoch progress bar with average loss and current weights
        concept_weights_str = ', '.join([f'{w:.3f}' for w in concept_weights.detach().cpu().numpy()])
        epoch_pbar.set_postfix({
            'Avg Loss': f'{avg_loss:.6f}',
            'Weights': f'[{concept_weights_str}]'
        })
    
    print("Concept weight optimization completed!")
    print(f"Final concept weights: {concept_weights.detach().cpu().numpy()}")
    
    # Save the learned concept weights
    output_path = '../output/save_data/learned_concept_weights.npy'
    np.save(output_path, concept_weights.detach().cpu().numpy())
    print(f"Learned concept weights saved to: {output_path}")
    
    return concept_weights.detach().cpu().numpy()

print("Loading new concept dataset...")
#load the new concept dataset
new_dataset = PushTImageDataset(['../output/save_data/bump.pkl'], [0], 
                                pred_horizon=pred_horizon, obs_horizon=obs_horizon, action_horizon=action_horizon)

#extract actions from the dataset for visualization
print(f"Dataset size: {len(new_dataset)}")
actions_list = []
for i in range(len(new_dataset)):
    sample = new_dataset[i]
    action = sample['action']  # shape: (pred_horizon, action_dim)
    actions_list.append(action)

# Stack actions into array with shape (N, pred_horizon, action_dim)
actions_array = np.stack(actions_list, axis=0)
print(f"Actions array shape: {actions_array.shape}")

#visualize the new concept dataset
visualize_trajectories(actions_array, n=20, gif_path='../output/save_data/new_concepts.gif')
if __name__ == "__main__":
    # Run the concept inference
    # learned_weights = infer_new_concepts(new_concept_dataset_path='../output/save_data/bump.pkl')
    # print("Concept learning completed. Learned weights:", learned_weights)
    # #save the learned weights
    # np.save('../output/save_data/learned_concept_weights.npy', learned_weights)
    # print("Learned weights saved to: ../output/save_data/learned_concept_weights.npy")
    pass

