import io
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib import rcParams
from PIL import Image
from diffusion_policy.utils.img_to_gif import np_to_img

def pad_and_stack_trajectories(trajectories):
    """
    trajectories: list of np.ndarray with shape (Ti, action_dim)
    returns: torch.Tensor of shape (N, T_max, action_dim)
    """
    # get max length
    max_len = max(traj.shape[0] for traj in trajectories)
    action_dim = trajectories[0].shape[1]
    
    padded = []
    for traj in trajectories:
        T = traj.shape[0]
        pad_len = max_len - T
        if pad_len > 0:
            # pad with last action
            pad = np.repeat(traj[-1:].copy(), pad_len, axis=0)
            padded_traj = np.concatenate([traj, pad], axis=0)
        else:
            padded_traj = traj
        padded.append(padded_traj)
    
    padded_array = np.stack(padded)  # (N, T_max, action_dim)
    return torch.tensor(padded_array, dtype=torch.float32)


def _fig_to_pil(fig):
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(img)

def visualize_trajectories(actions,
                            *,
                            n: int = 1,
                            gif_path: str | Path = "training_trajs.gif",
                            fps: int = 5,
                            seed: int | None = None,
                            background_img: str | Path | None = None,
                            dpi: int = 120):
    """
    Create an animated GIF that shows `n` sampled training trajectories unfolding over time.
    """
    rcParams["figure.dpi"] = dpi

    # ---- tensor → numpy -------------------------------------------------------
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()
    acts = np.asarray(actions)
    N, Tp, D = acts.shape
    if D != 2:
        raise ValueError(f"Expected last dim 2, got {D}")

    # ---- sample trajectories --------------------------------------------------
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(n, N), replace=False)
    trajs = acts[idx]  # (n, Tp, 2)

    # ---- color palette --------------------------------------------------------
    palette = plt.get_cmap("tab10", n)

    # ---- define fixed canvas size ---------------------------------------------
    figsize = (5, 5)
    xlim = (-1, 1)
    ylim = (-1, 1)
    frames = []
    for t in range(Tp):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        if background_img is not None:
            #check if background_img is a tensor
            if isinstance(background_img, torch.Tensor):
                background = background_img.permute(1,2,0).cpu().numpy()
                background = np_to_img(background)
            elif isinstance(background_img, np.ndarray):
                background = np_to_img(background_img)
            else:
                background = Image.open(background_img)
            ax.imshow(background, extent=[*xlim, *ylim], zorder=0)
        # ax.axis('off')  # remove axis ticks for consistent layout

        for k in range(n):
            ax.plot(trajs[k, :t + 1, 0],
                    trajs[k, :t + 1, 1],
                    color=palette(k),
                    linewidth=1.8,
                    alpha=0.8)
            ax.scatter(trajs[k, t, 0],
                       trajs[k, t, 1],
                       color=palette(k),
                       edgecolors="k",
                       s=70,
                       zorder=3)
        frames.append(_fig_to_pil(fig))

    # ---- save GIF -------------------------------------------------------------
    duration = 1.0 / fps
    #loop forever
    loop = 0
    imageio.mimsave(str(gif_path), frames, format="GIF", duration=duration, loop=loop)
    print(f"Saved GIF ➜  {gif_path}")

    return frames