import torch
import math
def sigmoid_beta_schedule(T, beta_min=1e-4, beta_max=0.02):
    t = torch.linspace(-3, 3, T)
    sig = torch.sigmoid(t)
    betas = beta_min + (beta_max - beta_min) * sig
    return betas
def cosine_beta_schedule(n_steps: int, s: float = 0.008) -> torch.Tensor:
    """Square‑cosine schedule from *Improved DDPM* (iDDPM).
    Args:
        n_steps: total diffusion steps T
        s: small offset to prevent β₁ from blowing up (default 0.008)
    Returns:
        (T,) tensor of β_t values in (0, 1)
    """
    ts = torch.linspace(0, n_steps, n_steps + 1, dtype=torch.float32) / n_steps
    alphas_cumprod = torch.cos((ts + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalise to 1 at t=0
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(max=0.999)