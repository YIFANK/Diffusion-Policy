import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def get_timestep_embedding(timesteps: torch.Tensor,
                           dim: int) -> torch.Tensor:
    """Transform scalar timesteps to sinusoidal embeddings (as in Transformer).
    Args:
        timesteps: (B,) int64 tensor of diffusion steps
        dim: embedding dimension (must be even)
    Returns:
        (B, dim) float tensor
    """
    half = dim // 2
    assert dim % 2 == 0, "Embedding dimension must be even"
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(timesteps.device)  # (half,)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
    return emb


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.seq(x)
    
class ImageEncoder(nn.Module):
    def __init__(self, out_dim=64):   # << 64 instead of 512
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.SiLU(),   # 96→48
            nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),  # 48→24
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),  # 24→12
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),  # -> 64
            nn.LayerNorm(64),                      # stabilise
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W)"""
        assert x.dim() == 4 and x.size(1) == 3, "Input must be (B, 1, H, W)"
        return self.net(x)


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        act_dim: int,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        enc_dim: int = 64,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.act_dim = act_dim
        self.n_steps = n_steps

        # β schedule & buffers
        betas = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_ac", torch.sqrt(ac))
        self.register_buffer("sqrt_1m_ac", torch.sqrt(1.0 - ac))

        # networks
        self.encoder = ImageEncoder(out_dim=enc_dim)
        self.eps_net = MLP(in_dim=act_dim + enc_dim + 128, out_dim=act_dim)

    # -------------------- diffusion helpers -------------------- #
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        s_ac = self.sqrt_ac[t][:, None]
        s_1m = self.sqrt_1m_ac[t][:, None]
        return s_ac * x0 + s_1m * noise

    # --------------------------- forward (training) -------------------------- #
    def forward(self, obs_img: torch.Tensor, actions: torch.Tensor):
        """obs_img: (B,3,H,W), actions: (B,act_dim)"""
        B = actions.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=actions.device)
        noise = torch.randn_like(actions)
        x_t = self.q_sample(actions, t, noise)
        cond = self.encoder(obs_img)  # (B, enc_dim)
        t_emb = get_timestep_embedding(t, 128)
        eps_in = torch.cat([x_t, cond, t_emb], dim=-1)
        eps_pred = self.eps_net(eps_in)
        return F.mse_loss(eps_pred, noise)

    # --------------------------- reverse step ------------------------------- #
    @torch.no_grad()
    def p_sample(self, x_t, t: int, cond):
        beta_t = self.betas[t]
        sqrt_recip_alpha = 1.0 / math.sqrt(1.0 - beta_t)
        sqrt_1m_ac_t = self.sqrt_1m_ac[t]

        t_emb = get_timestep_embedding(torch.full((x_t.size(0),), t, device=x_t.device), 128)
        eps_in = torch.cat([x_t, cond, t_emb], dim=-1)
        eps_pred = self.eps_net(eps_in)

        mean = sqrt_recip_alpha * (x_t - beta_t / sqrt_1m_ac_t * eps_pred)
        if t > 0:
            x_prev = mean + math.sqrt(beta_t) * torch.randn_like(x_t)
        else:
            x_prev = mean
        return x_prev

    # --------------------------- sampling API ------------------------------- #
    @torch.no_grad()
    def sample(self, obs_img: torch.Tensor, n: int = 1):
        """obs_img: (3,H,W) or (B,3,H,W)"""
        if obs_img.dim() == 3:
            obs_img = obs_img.unsqueeze(0)
        cond = self.encoder(obs_img).repeat(n, 1)
        x_t = torch.randn(n, self.act_dim, device=obs_img.device)
        for t in reversed(range(self.n_steps)):
            x_t = self.p_sample(x_t, t, cond)
        return x_t
    



