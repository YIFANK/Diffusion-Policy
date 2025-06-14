import io
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib import rcParams
from PIL import Image


def _fig_to_pil(fig) -> Image.Image:
    """Convert a Matplotlib figure to a PIL image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


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
    palette = plt.cm.get_cmap("tab10", n)

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
    imageio.mimsave(str(gif_path), frames, format="GIF", duration=duration)
    print(f"Saved GIF ➜  {gif_path}")

    return frames