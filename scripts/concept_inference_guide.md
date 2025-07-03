# Concept Inference for Diffusion Policy and Flow Matching

This guide explains how to use the extended concept inference framework that supports both diffusion policies and flow matching policies.

## Overview

The concept inference framework learns to combine existing concepts to model new behaviors. Given:
- A frozen pre-trained policy (diffusion or flow matching)
- A set of base concept embeddings 
- Demonstrations of a new concept

The framework learns optimal weights to combine the base concepts to represent the new concept.

## Key Files

1. **`concept_inference.py`** - Main concept learning script (supports both policy types)
2. **`compare_concept_inference.py`** - Comparative analysis between diffusion and flow matching
3. **`concept_inference_guide.md`** - This usage guide

## Basic Usage

### 1. Single Policy Concept Learning

```python
from concept_inference import infer_new_concepts

# For diffusion policy
results_diffusion = infer_new_concepts(
    model_path='../output/diffusion_policy.pth',
    policy_type="diffusion",  # Key parameter!
    new_concept_dataset_path='../output/save_data/new_concept.pkl',
    weights_output_path='../output/concepts/weights_diffusion.npy',
    embeddings_output_path='../output/concepts/embeddings_diffusion.npy',
    num_epochs=500,
    learning_rate=0.0003,
    batch_size=64,
    K=1  # Number of concepts to learn
)

# For flow matching policy  
results_flow = infer_new_concepts(
    model_path='../output/flow_policy.pth',
    policy_type="flow_matching",  # Key parameter!
    new_concept_dataset_path='../output/save_data/new_concept.pkl',
    weights_output_path='../output/concepts/weights_flow.npy',
    embeddings_output_path='../output/concepts/embeddings_flow.npy',
    num_epochs=500,
    learning_rate=0.0003,
    batch_size=64,
    K=1
)
```

### 2. Comparative Analysis

```python
from compare_concept_inference import compare_concept_inference

# Compare both policy types
results = compare_concept_inference(
    diffusion_model_path='../output/diffusion_policy.pth',
    flow_model_path='../output/flow_policy.pth',
    new_concept_dataset_path='../output/save_data/new_concept.pkl',
    output_dir='../output/comparison/',
    num_epochs=200,
    learning_rate=0.0003,
    batch_size=64,
    K=1,
    num_runs=3  # Multiple runs for robustness
)
```

## Key Differences: Diffusion vs Flow Matching

### Mathematical Formulation

**Diffusion Policy Loss:**
```
E[||ε - (ε_θ(x_t, c_0, s_0, t) + Σ ω_k(ε_θ(x_t, c̃_k, s_0, t) - ε_θ(x_t, c_0, s_0, t)))||²]
```

**Flow Matching Loss:**
```
E[||v - (v_θ(x_t, c_0, s_0, t) + Σ ω_k(v_θ(x_t, c̃_k, s_0, t) - v_θ(x_t, c_0, s_0, t)))||²]
```

Where:
- `ε` = noise, `v` = vector field  
- `ε_θ` = noise predictor, `v_θ` = vector field predictor
- `x_t` = noisy sample at time t
- `c_k` = concept embeddings, `ω_k` = concept weights
- `s_0` = observations

### Implementation Differences

| Aspect | Diffusion | Flow Matching |
|--------|-----------|---------------|
| **Time sampling** | Discrete steps (0-1000) | Continuous time (0-1) |
| **Forward process** | DDPM noise schedule | Linear interpolation |
| **Target** | Predict noise `ε` | Predict vector field `v` |
| **True target** | Added noise | `data - noise` |
| **Sampling** | DDIM/DDPM steps | ODE integration |

## Parameters

### Core Parameters

- **`policy_type`**: `"diffusion"` or `"flow_matching"`
- **`model_path`**: Path to saved policy model
- **`new_concept_dataset_path`**: Path to demonstrations of new concept
- **`K`**: Number of concept embeddings to learn
- **`num_epochs`**: Training epochs (typically 500-1000)
- **`learning_rate`**: Learning rate (typically 0.0003)

### Output Files

- **Weights**: `weights_output_path` - Learned concept combination weights
- **Embeddings**: `embeddings_output_path` - Learned concept embeddings
- **Logs**: Wandb logs with training progress and visualizations

## Advanced Usage

### Multiple Concepts

```python
# Learn multiple concepts simultaneously
results = infer_new_concepts(
    # ... other parameters ...
    K=3,  # Learn 3 concept embeddings
    learning_rate=0.0001  # Lower LR for stability with more concepts
)
```

### Custom Datasets

```python
# Your dataset should return batches with:
# - 'image': (B, obs_horizon, C, H, W) 
# - 'agent_pos': (B, obs_horizon, 2)
# - 'action': (B, pred_horizon, action_dim)
# - 'text': List of text descriptions (optional)

from diffusion_policy.data.dataset import PushTImageDataset

dataset = PushTImageDataset(
    [path_to_demonstrations],
    [concept_label],
    pred_horizon=16,
    obs_horizon=2, 
    action_horizon=8
)
```

### Hyperparameter Guidelines

| Parameter | Diffusion | Flow Matching | Notes |
|-----------|-----------|---------------|-------|
| **Learning Rate** | 0.0003 | 0.0003 | Flow may need lower LR |
| **Batch Size** | 64 | 64 | Larger for stability |
| **Epochs** | 500-1000 | 500-1000 | Flow may converge faster |
| **K (concepts)** | 1-3 | 1-3 | More concepts = harder optimization |

## Output Analysis

### Training Progress

Monitor these metrics:
- **Loss convergence**: Should decrease and stabilize
- **Weight values**: Final learned combination weights
- **Embedding norms**: Magnitude of learned embeddings
- **UMAP visualization**: Embedding evolution in 2D space

### Interpretation

**Good Results:**
- Loss converges to low value
- Weights are stable across runs
- Embeddings have reasonable norms (1-10)
- UMAP shows clear separation from base concepts

**Poor Results:**
- Loss doesn't converge or oscillates
- Weights vary significantly across runs  
- Very large or very small embedding norms
- Embeddings collapse to base concepts in UMAP

## Troubleshooting

### Common Issues

1. **Loss not decreasing**
   - Check dataset format and loading
   - Reduce learning rate
   - Verify model loads correctly

2. **Unstable training**
   - Lower learning rate
   - Reduce batch size
   - Decrease number of concepts K

3. **Poor concept separation**
   - Increase number of epochs
   - Try different K values
   - Check quality of demonstration data

4. **Memory issues**
   - Reduce batch size
   - Use fewer concepts (lower K)
   - Enable gradient checkpointing

### Performance Tips

- **GPU utilization**: Use larger batch sizes when memory allows
- **Convergence**: Flow matching often converges faster than diffusion
- **Stability**: Multiple random seed runs recommended for robust results
- **Visualization**: UMAP plots help verify concept learning progress

## Example Workflows

### Quick Test
```bash
python concept_inference.py  # Runs with default parameters
```

### Full Comparison Study
```bash
python compare_concept_inference.py  # Compares both policy types
```

### Custom Experiment
```python
# Custom script combining both approaches
from concept_inference import infer_new_concepts

for policy_type in ["diffusion", "flow_matching"]:
    for lr in [0.0001, 0.0003, 0.001]:
        results = infer_new_concepts(
            policy_type=policy_type,
            learning_rate=lr,
            # ... other params ...
        )
        # Analyze results...
```

This framework provides a unified approach to concept learning across different generative policy architectures, enabling systematic comparison and analysis of concept inference capabilities. 