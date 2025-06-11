# policy.py (replace old DiffusionPolicy definition)
import math, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18
from FiLM import EpsNet, sinusoidal_emb
from Resnet import ResNetEncoder
from noise_schedule import cosine_beta_schedule,sigmoid_beta_schedule
def ddim_step(policy, x_t, t, cond, eta: float = 0.0):
    # ---- gather schedule values --------------------------------------------
    betas        = policy.betas
    sqrt_ac      = policy.sqrt_ac            # √ᾱₜ
    sqrt_1m_ac   = policy.sqrt_1m_ac         # √(1−ᾱₜ)
    alpha_bar_t  = sqrt_ac[t] ** 2
    alpha_bar_tm1 = sqrt_ac[t - 1] ** 2 if t > 0 else sqrt_ac[0] ** 2

    # ---- predict ε_θ(x_t, cond) -------------------------------------------
    eps = policy.eps_net(x_t.permute(0, 2, 1), cond)      # (B, ad, Tp)
    eps = eps.permute(0, 2, 1)                            # (B, Tp, ad)

    # ---- compute x0 prediction ---------------------------------------------
    x0_pred = (x_t - sqrt_1m_ac[t] * eps) / sqrt_ac[t]

    # ---- deterministic part -------------------------------------------------
    coeff_1 = math.sqrt(alpha_bar_tm1)
    coeff_2 = math.sqrt(1 - alpha_bar_tm1)
    dir_xt  = eps * coeff_2
    x_prev  = coeff_1 * x0_pred + dir_xt

    # ---- optional noise (stochastic DDIM) -----------------------------------
    if eta > 0 and t > 0:
        sigma_t = eta * math.sqrt(
            (1 - alpha_bar_tm1) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_tm1)
        )
        noise   = torch.randn_like(x_t)
        x_prev  += sigma_t * noise

    return x_prev

class DiffusionPolicy(nn.Module):
    def __init__(self,
                 act_dim: int,
                 Tp: int = 8,
                 To: int = 2,
                 n_steps: int = 1000,
                 hidden: int = 128,
                 K: int = 4,                     # FiLM blocks
                 enc_dim: int = 512,
                 schedule: str = "linear",
                 device: str = "cpu"):
        super().__init__()
        self.Tp, self.act_dim, self.device = Tp, act_dim, device

        # β-schedule
        betas = torch.linspace(1e-4, 2e-2, n_steps, dtype=torch.float32)
        if schedule == "cosine":
            betas = cosine_beta_schedule(n_steps)
        elif schedule == "sigmoid":
            betas = sigmoid_beta_schedule(n_steps)
        alphas = 1. - betas
        ac = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_ac", torch.sqrt(ac))
        self.register_buffer("sqrt_1m_ac", torch.sqrt(1. - ac))

        # nets
        self.encoder = ResNetEncoder(enc_dim)
        self.eps_net = EpsNet(act_dim, hidden,
                        cond_dim=enc_dim * To + 128,  # concat, not mean
                        K=K)

    # -------------- helpers --------------------------------------------------
    def prepare_cond(self, obs, t):               # obs:(B,To,3,H,W)
        B, To = obs.size(0), obs.size(1)
        obs   = obs.view(B * To, *obs.shape[2:])  # (B·To,3,H,W)
        feat  = self.encoder(obs).view(B, To, -1) # (B,To, enc_dim)
        feat  = feat.reshape(B, To * feat.size(-1))         # (B,To·enc_dim)
        t_emb = sinusoidal_emb(t, 128)                         # (B,128)
        return torch.cat([feat, t_emb], 1)                     # (B,enc+128)

    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        s_ac  = self.sqrt_ac[t][:, None, None]
        s_1m  = self.sqrt_1m_ac[t][:, None, None]
        return s_ac * x0 + s_1m * noise

    # -------------- training -------------------------------------------------
    def forward(self, obs, actions):               # obs:(B,To,3,H,W)  act:(B,Tp,ad)
        B, T, _ = actions.shape
        t    = torch.randint(0, len(self.betas), (B,), device=actions.device)
        noise = torch.randn_like(actions)
        x_t  = self.q_sample(actions, t, noise)    # (B,Tp,ad)
        x_t  = x_t.permute(0, 2, 1)                # (B,ad,L)
        cond = self.prepare_cond(obs, t)           # (B,cond_dim)
        eps  = self.eps_net(x_t, cond)             # (B,ad,L)
        return F.mse_loss(eps.permute(0, 2, 1), noise)

    # -------------- sampling -------------------------------------------------
    @torch.no_grad()
    def p_step(self, x_t, t, cond):
        beta_t, sqrt_1m = self.betas[t], self.sqrt_1m_ac[t]
        sqrt_inv = 1. / math.sqrt(1. - beta_t)
        L = x_t.size(1)
        eps = self.eps_net(x_t.permute(0, 2, 1), cond)        # (B,ad,L)
        eps = eps.permute(0, 2, 1)
        mean = sqrt_inv * (x_t - beta_t / sqrt_1m * eps)
        if t:
            noise = torch.randn_like(x_t)
            x_prev = mean + math.sqrt(beta_t) * noise
        else:
            x_prev = mean
        return x_prev

    @torch.no_grad()
    def sample(self, obs, n=5):
        # obs:(To,3,H,W)  →  repeat on batch dim
        print(f"Sampling {n} trajectories with Tp={self.Tp} and observation shape={obs.shape}...")
        obs = obs.unsqueeze(0).repeat(n, 1, 1, 1, 1)
        cond = self.prepare_cond(obs, torch.zeros(n, dtype=torch.long, device=obs.device))
        x_t  = torch.randn(n, self.Tp, self.act_dim, device=self.device)
        for t in reversed(range(len(self.betas))):
            x_t = self.p_step(x_t, t, cond)
        return x_t                                    # (n,Tp,act_dim)
    
    @torch.no_grad()
    def sample_ddim(self, obs, n=5, steps=100, eta=0):
        """
        DDIM sampling interface.
        • `steps` : # inference steps (≤ self.betas.size(0))
        • `eta`   : stochasticity (0 = deterministic DDIM).
        """
        # 1) prep cond exactly as before
        if obs.dim() == 4:
            obs = obs.unsqueeze(0)                   # (1,To,3,H,W)
        obs   = obs.to(self.device).repeat(n, 1, 1, 1, 1)
        cond  = self.prepare_cond(obs,
            torch.zeros(n, dtype=torch.long, device=self.device))

        # 2) choose an evenly-spaced timestep subset
        T_total  = len(self.betas)
        schedule = torch.linspace(T_total - 1, 0, steps, dtype=torch.long)
        x_t = torch.randn(n, self.Tp, self.act_dim, device=self.device)

        # 3) loop
        for i, t in enumerate(schedule):
            x_t = ddim_step(self, x_t, int(t), cond, eta=eta)

        return x_t

