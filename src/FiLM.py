# film_blocks.py
import math, torch, torch.nn as nn, torch.nn.functional as F

# ---------- helpers ----------------------------------------------------------
def sinusoidal_emb(t, dim):                        # (B,) → (B,dim)
    device, half = t.device, dim // 2
    freqs = torch.exp(-math.log(10000) *
                      torch.arange(half, device=device) / half)
    ang = t.float().unsqueeze(1) * freqs           # (B,half)
    return torch.cat([ang.sin(), ang.cos()], -1)   # (B,dim)

# ---------- FiLM blocks ------------------------------------------------------
class FiLMedResBlock(nn.Module):
    """
    Conv->BN->SiLU, then FiLM:   y = a * x + b
    a,b are produced from the *conditioning vector* (B,cond_dim).
    """
    def __init__(self, channels: int, cond_dim: int, k_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, k_size, padding=k_size//2)
        self.bn   = nn.BatchNorm1d(channels)
        self.act  = nn.SiLU()
        # linear → 2C so we get a and b
        self.to_film = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x, cond):                    # x:(B,C,L)  cond:(B,cond_dim)
        a, b = self.to_film(cond).chunk(2, dim=1)  # both (B,C)
        a, b = a.unsqueeze(-1), b.unsqueeze(-1)    # (B,C,1)
        out  = self.act(self.bn(self.conv(x)))
        return a * out + b                        # FiLM-modulated residual

# ---------- ε-network --------------------------------------------------------
class EpsNet(nn.Module):
    """
    Projects action sequence to hidden dim, applies K FiLMed blocks, and
    projects back to action dim.
    """
    def __init__(self,
                 act_dim: int,
                 hidden: int,
                 cond_dim: int,
                 K: int = 4):                     # K = #FiLM blocks
        super().__init__()
        self.in_proj  = nn.Conv1d(act_dim, hidden, 1)
        self.blocks   = nn.ModuleList(
            [FiLMedResBlock(hidden, cond_dim) for _ in range(K)]
        )
        self.out_proj = nn.Conv1d(hidden, act_dim, 1)

    def forward(self, x, cond):                   # x:(B,act_dim,L)
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x, cond) + x                  # simple residual
        return self.out_proj(x)                   # (B,act_dim,L)
