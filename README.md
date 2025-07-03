# Diffusion Policy Implementation

This repository contains an implementation of **Diffusion Policy** for robotic manipulation tasks, specifically demonstrated on a 2D pushing task in a simulated environment. The project implements the core concepts from the paper "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" using PyTorch and a custom toy environment.

## 🎯 Overview

Diffusion policies use denoising diffusion models to generate robot actions conditioned on visual observations. This implementation features:

- **Vision-based control**: Uses ResNet18 encoder for image observations
- **Text conditioning**: Supports conditional action generation with simple text embeddings
- **Classifier-free guidance**: Enables controllable action generation
- **2D pushing task**: Demonstrated on a simulated environment with objects to push

## 🏗️ Project Structure

```
Diffusion-Policy/
├── diffusion_policy/           # Main Python package
│   ├── __init__.py            # Package initialization
│   ├── models/                # Model implementations
│   │   ├── __init__.py
│   │   ├── diffusion_policy.py  # Core diffusion policy
│   │   ├── network.py         # UNet architecture
│   │   └── vision_encoder.py  # Vision encoding utilities
│   ├── data/                  # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── dataset.py         # Dataset implementations
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── visualization.py   # Trajectory visualization
│   │   └── img_to_gif.py      # Image processing
│   └── configs/               # Configuration management
│       ├── __init__.py
│       └── config.yaml        # Default configuration
├── scripts/                   # Executable scripts
│   ├── train.py              # Training script
│   ├── inference.py          # Inference and evaluation
│   └── concept_inference.py  # Concept-based inference
├── examples/                  # Example notebooks and demos
│   └── diffusion_policy_vision_pusht_demo.ipynb
├── tests/                     # Test suite
│   ├── __init__.py
│   └── test_diffusion_policy.py
├── output/                    # Generated outputs
│   ├── save_data/            # Training datasets
│   └── *.gif                 # Generated visualizations
├── requirements.txt           # Package dependencies
├── pyproject.toml            # Package configuration
├── .gitignore                # Git ignore patterns
└── README.md                 # This file
```

## 🚀 Features

### Core Components

1. **DiffusionPolicy** (`dp.py`): Main policy class implementing DDPM-based action generation
2. **ConditionalUnet1D** (`network.py`): 1D U-Net for noise prediction with FiLM conditioning
3. **PushTImageDataset** (`Dataset.py`): Dataset loader with normalization and episode handling
4. **Vision Encoder** (`vision_encoder.py`): ResNet18 with GroupNorm for stable training

### Key Capabilities

- **Multi-modal conditioning**: Vision + low-dimensional state + text
- **Action sequence prediction**: Generates action horizons (default: 8 steps)
- **Observation history**: Uses observation horizons (default: 2 steps)
- **EMA training**: Exponential moving average for stable training
- **Trajectory visualization**: Automatic GIF generation of predicted trajectories

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Install from source

```bash
# Clone the repository
git clone <repository-url>
cd Diffusion-Policy

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Install dependencies only

```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Quick Start

```python
from diffusion_policy import DiffusionPolicy, PushTImageDataset
from diffusion_policy.configs import get_default_config

# Load configuration
config = get_default_config()

# Create model
policy = DiffusionPolicy(
    obs_horizon=config.model.obs_horizon,
    pred_horizon=config.model.pred_horizon,
    action_dim=config.model.action_dim,
    num_diffusion_iters=config.model.num_diffusion_iters
)

# Load dataset
dataset = PushTImageDataset(
    dataset_paths=[config.data.left_path, config.data.right_path],
    text_conditions=[-1, 1],
    pred_horizon=config.model.pred_horizon,
    obs_horizon=config.model.obs_horizon,
    action_horizon=config.model.action_horizon
)
```

### Training

```bash
# From the repository root
cd scripts
python train.py

# Or with custom config
python train.py --config path/to/your/config.yaml
```

### Inference/Evaluation

```bash
cd scripts
python inference.py

# With custom model and settings
python inference.py --model_path ../output/your_model.pth --num_episodes 20
```

**Evaluation Features:**
- Tests trained policy on random environments
- Generates trajectory visualizations
- Supports conditional and unconditional generation
- Collision detection and reward computation

### Data Generation

The project uses trajectory data from a 2D pushing environment. Training data should be placed in:
- `output/save_data/left.pkl` - Left movement demonstrations
- `output/save_data/right.pkl` - Right movement demonstrations
- `output/save_data/embeddings.npy` - Text embeddings

## 🔧 Configuration

### Key Hyperparameters

```python
obs_horizon = 2      # Number of observations to stack
pred_horizon = 16    # Number of actions to predict
action_dim = 2       # Action dimension (x, y movement)
action_horizon = 8   # Number of actions to execute
```

### Model Architecture

- **Vision Encoder**: ResNet18 (512 features) + GroupNorm
- **Noise Prediction**: 1D U-Net with FiLM conditioning
- **Text Encoder**: Simple embedding (64 dimensions)
- **Scheduler**: DDPM with squared cosine beta schedule

## 📊 Monitoring

The training script integrates with **Weights & Biases** for experiment tracking:

- Loss curves
- Generated trajectory visualizations
- Hyperparameter logging
- Model performance metrics

## 🎨 Visualization

The project automatically generates trajectory visualizations:

- **Conditional trajectories**: `output/cond_trajectories.gif`
- **Unconditional trajectories**: `output/uncond_trajectories.gif`
- **Evaluation trajectories**: Per-episode GIFs during evaluation

## 🧪 Environment

The implementation uses a custom 2D pushing environment with:
- **Agent**: Controllable agent with position feedback
- **Objects**: Pushable objects (boxes, circles)
- **Goal**: Navigate agent to target region (450, 450)
- **Observations**: 96x96 RGB images + agent position
- **Actions**: 2D continuous movement commands

## 📈 Results

The trained policy demonstrates:
- Vision-based navigation in 2D environments
- Text-conditioned action generation (left/right)
- Smooth trajectory generation through diffusion sampling
- Collision avoidance behavior

## 🔬 Technical Details

### Diffusion Process

1. **Forward Process**: Add Gaussian noise to action sequences
2. **Reverse Process**: Learn to denoise actions conditioned on observations
3. **Sampling**: Use DDPM scheduler for iterative denoising
4. **Guidance**: Classifier-free guidance for conditional generation

### Training Process

1. Load demonstration data with text labels
2. Sample noise and timesteps for diffusion training
3. Predict noise using conditional U-Net
4. Optimize MSE loss between predicted and actual noise
5. Update EMA model for stable inference

## 🤝 Contributing

This is a research implementation. Contributions welcome for:
- Additional environment support
- Architecture improvements
- Better visualization tools
- Documentation enhancements

## 📚 References

- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)

## 📄 License

[Add your license information here]