# Compare concept inference between diffusion and flow matching policies
import sys
import os
# Add parent directory to path so we can import diffusion_policy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from concept_inference import infer_new_concepts
import wandb
from pathlib import Path

def compare_concept_inference(
    diffusion_model_path: str = '../output/diffusion_policy.pth',
    flow_model_path: str = '../output/flow_policy.pth',
    new_concept_dataset_path: str = '../output/save_data/bump.pkl',
    output_dir: str = '../output/comparison/',
    num_epochs: int = 500,
    learning_rate: float = 0.0003,
    batch_size: int = 64,
    K: int = 1,
    num_runs: int = 3  # Number of random seeds to average over
):
    """
    Compare concept learning between diffusion and flow matching policies.
    
    Args:
        diffusion_model_path: Path to saved diffusion policy model
        flow_model_path: Path to saved flow matching policy model
        new_concept_dataset_path: Path to dataset with new concept demonstrations
        output_dir: Directory to save comparison results
        num_epochs: Number of training epochs per run
        learning_rate: Learning rate for concept weight optimization
        batch_size: Batch size for training
        K: Number of concepts to learn
        num_runs: Number of random initialization runs for robustness
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Store results for both policy types
    results = {
        'diffusion': {'weights': [], 'embeddings': [], 'final_losses': []},
        'flow_matching': {'weights': [], 'embeddings': [], 'final_losses': []}
    }
    
    print(f"Starting comparative concept inference study with {num_runs} runs each...")
    print(f"Epochs per run: {num_epochs}, Learning rate: {learning_rate}, Batch size: {batch_size}")
    
    # Run concept inference for both policy types across multiple seeds
    for run_idx in range(num_runs):
        print(f"\n=== Run {run_idx + 1}/{num_runs} ===")
        
        # Set random seed for reproducibility
        torch.manual_seed(42 + run_idx)
        np.random.seed(42 + run_idx)
        
        # Run for diffusion policy
        print(f"Running diffusion policy (seed {42 + run_idx})...")
        diffusion_result = infer_new_concepts(
            model_path=diffusion_model_path,
            policy_type="diffusion",
            new_concept_dataset_path=new_concept_dataset_path,
            weights_output_path=f'{output_dir}/diffusion_weights_run{run_idx}.npy',
            embeddings_output_path=f'{output_dir}/diffusion_embeddings_run{run_idx}.npy',
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            K=K
        )
        results['diffusion']['weights'].append(diffusion_result['weights'])
        results['diffusion']['embeddings'].append(diffusion_result['embeddings'])
        
        # Reset random seed for fair comparison
        torch.manual_seed(42 + run_idx)  
        np.random.seed(42 + run_idx)
        
        # Run for flow matching policy
        print(f"Running flow matching policy (seed {42 + run_idx})...")
        flow_result = infer_new_concepts(
            model_path=flow_model_path,
            policy_type="flow_matching", 
            new_concept_dataset_path=new_concept_dataset_path,
            weights_output_path=f'{output_dir}/flow_weights_run{run_idx}.npy',
            embeddings_output_path=f'{output_dir}/flow_embeddings_run{run_idx}.npy',
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            K=K
        )
        results['flow_matching']['weights'].append(flow_result['weights'])
        results['flow_matching']['embeddings'].append(flow_result['embeddings'])
    
    # Analyze and visualize results
    print("\n=== Analysis Results ===")
    analyze_results(results, output_dir, K)
    
    return results

def analyze_results(results, output_dir, K):
    """
    Analyze and visualize the comparison results.
    """
    
    # Convert lists to numpy arrays for easier analysis
    diffusion_weights = np.array(results['diffusion']['weights'])  # (num_runs, K)
    flow_weights = np.array(results['flow_matching']['weights'])   # (num_runs, K)
    
    diffusion_embeddings = np.array(results['diffusion']['embeddings'])  # (num_runs, K, embedding_dim)
    flow_embeddings = np.array(results['flow_matching']['embeddings'])   # (num_runs, K, embedding_dim)
    
    print(f"Diffusion weights shape: {diffusion_weights.shape}")
    print(f"Flow matching weights shape: {flow_weights.shape}")
    print(f"Diffusion embeddings shape: {diffusion_embeddings.shape}")
    print(f"Flow matching embeddings shape: {flow_embeddings.shape}")
    
    # 1. Weight Statistics
    print("\n--- Weight Analysis ---")
    diff_mean_weights = np.mean(diffusion_weights, axis=0)
    diff_std_weights = np.std(diffusion_weights, axis=0)
    flow_mean_weights = np.mean(flow_weights, axis=0)
    flow_std_weights = np.std(flow_weights, axis=0)
    
    print(f"Diffusion weights: {diff_mean_weights} ± {diff_std_weights}")
    print(f"Flow weights: {flow_mean_weights} ± {flow_std_weights}")
    
    # 2. Embedding Statistics  
    print("\n--- Embedding Analysis ---")
    diff_embedding_norms = np.linalg.norm(diffusion_embeddings, axis=2)  # (num_runs, K)
    flow_embedding_norms = np.linalg.norm(flow_embeddings, axis=2)       # (num_runs, K)
    
    print(f"Diffusion embedding norms: {np.mean(diff_embedding_norms, axis=0)} ± {np.std(diff_embedding_norms, axis=0)}")
    print(f"Flow embedding norms: {np.mean(flow_embedding_norms, axis=0)} ± {np.std(flow_embedding_norms, axis=0)}")
    
    # 3. Cross-run consistency (how similar are results across runs?)
    print("\n--- Consistency Analysis ---")
    diff_weight_consistency = np.std(diffusion_weights, axis=0) / (np.abs(np.mean(diffusion_weights, axis=0)) + 1e-8)
    flow_weight_consistency = np.std(flow_weights, axis=0) / (np.abs(np.mean(flow_weights, axis=0)) + 1e-8)
    
    print(f"Diffusion weight coefficient of variation: {diff_weight_consistency}")
    print(f"Flow weight coefficient of variation: {flow_weight_consistency}")
    
    # 4. Create visualization plots
    create_comparison_plots(results, output_dir, K)
    
    # 5. Save summary statistics
    summary = {
        'diffusion': {
            'mean_weights': diff_mean_weights.tolist(),
            'std_weights': diff_std_weights.tolist(),
            'mean_embedding_norms': np.mean(diff_embedding_norms, axis=0).tolist(),
            'std_embedding_norms': np.std(diff_embedding_norms, axis=0).tolist(),
            'weight_cv': diff_weight_consistency.tolist()
        },
        'flow_matching': {
            'mean_weights': flow_mean_weights.tolist(),
            'std_weights': flow_std_weights.tolist(),
            'mean_embedding_norms': np.mean(flow_embedding_norms, axis=0).tolist(),
            'std_embedding_norms': np.std(flow_embedding_norms, axis=0).tolist(),
            'weight_cv': flow_weight_consistency.tolist()
        }
    }
    
    import json
    with open(f'{output_dir}/comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")

def create_comparison_plots(results, output_dir, K):
    """
    Create visualization plots comparing the two policy types.
    """
    
    diffusion_weights = np.array(results['diffusion']['weights'])
    flow_weights = np.array(results['flow_matching']['weights'])
    diffusion_embeddings = np.array(results['diffusion']['embeddings'])
    flow_embeddings = np.array(results['flow_matching']['embeddings'])
    
    # Set up the plotting style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            pass  # Use default style if seaborn not available
    sns.set_palette("husl")
    
    # 1. Weight Distribution Comparison
    fig, axes = plt.subplots(1, K, figsize=(5*K, 4))
    if K == 1:
        axes = [axes]
    
    for k in range(K):
        ax = axes[k]
        
        # Box plots for weight distributions
        data_to_plot = [diffusion_weights[:, k], flow_weights[:, k]]
        bp = ax.boxplot(data_to_plot, labels=['Diffusion', 'Flow Matching'], patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral'] 
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Learned Weight Value')
        ax.set_title(f'Weight Distribution - Concept {k+1}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/weight_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Embedding Norm Comparison
    diff_norms = np.linalg.norm(diffusion_embeddings, axis=2)
    flow_norms = np.linalg.norm(flow_embeddings, axis=2)
    
    fig, axes = plt.subplots(1, K, figsize=(5*K, 4))
    if K == 1:
        axes = [axes]
        
    for k in range(K):
        ax = axes[k]
        
        data_to_plot = [diff_norms[:, k], flow_norms[:, k]]
        bp = ax.boxplot(data_to_plot, labels=['Diffusion', 'Flow Matching'], patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            
        ax.set_ylabel('Embedding Norm')
        ax.set_title(f'Embedding Norm Distribution - Concept {k+1}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/embedding_norm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Weight vs Embedding Norm Scatter Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    for k in range(K):
        # Diffusion points
        ax.scatter(diffusion_weights[:, k], diff_norms[:, k], 
                  alpha=0.7, s=100, label=f'Diffusion - Concept {k+1}', 
                  marker='o', edgecolors='black', linewidth=0.5)
        
        # Flow matching points  
        ax.scatter(flow_weights[:, k], flow_norms[:, k],
                  alpha=0.7, s=100, label=f'Flow - Concept {k+1}',
                  marker='^', edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Learned Weight Value')
    ax.set_ylabel('Embedding Norm')
    ax.set_title('Weight vs Embedding Norm Relationship')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/weight_vs_norm_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Convergence Stability (Coefficient of Variation)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    # Calculate coefficient of variation for weights
    diff_cv = np.std(diffusion_weights, axis=0) / (np.abs(np.mean(diffusion_weights, axis=0)) + 1e-8)
    flow_cv = np.std(flow_weights, axis=0) / (np.abs(np.mean(flow_weights, axis=0)) + 1e-8)
    
    x = np.arange(K)
    width = 0.35
    
    rects1 = ax.bar(x - width/2, diff_cv, width, label='Diffusion', alpha=0.8, color='lightblue')
    rects2 = ax.bar(x + width/2, flow_cv, width, label='Flow Matching', alpha=0.8, color='lightcoral')
    
    ax.set_xlabel('Concept Index')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Learning Stability Comparison\n(Lower = More Stable)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Concept {i+1}' for i in range(K)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/stability_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {output_dir}")

if __name__ == "__main__":
    # Run comparison study
    results = compare_concept_inference(
        diffusion_model_path='../output/diffusion_policy.pth',
        flow_model_path='../output/flow_policy.pth', 
        new_concept_dataset_path='../output/save_data/bump.pkl',
        output_dir='../output/comparison/',
        num_epochs=200,  # Reduced for faster comparison
        learning_rate=0.0003,
        batch_size=64,
        K=1,
        num_runs=3
    )
    
    print("\nComparison study completed!")
    print("Check ../output/comparison/ for detailed results and visualizations.") 