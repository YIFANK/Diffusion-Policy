import numpy as np
import torch
import matplotlib.pyplot as plt


def visualize_trajectories(trajs,
                           *,
                           as_displacement: bool = True,
                           cmap: str = "viridis",
                           dot_size: int = 60,
                           line_alpha: float = 0.4,
                           ax=None):
    """
    Visualize N sampled 2-D trajectories.

    Parameters
    ----------
    trajs : (N, T, 2) torch.Tensor | np.ndarray
        The raw output of `policy.sample(...)`.
    as_displacement : bool, default True
        If True, each action is interpreted as a displacement; the
        function plots the cumulative sum so that the agent starts at (0,0).
        If False, points are taken as absolute coordinates already.
    cmap : str, default "viridis"
        Any Matplotlib colormap name.
    dot_size : int, default 60
        Size of the scatter dots.
    line_alpha : float, default 0.4
        Transparency of the grey path connecting the dots.
    ax : matplotlib.axes.Axes, optional
        Pass an existing axis; otherwise a new figure is created.

    Example
    -------
    >>> samples = policy.sample(obs, n=8)          # (8, Tp, 2)
    >>> visualize_trajectories(samples)
    """
    # ---------- tensor → ndarray ------------------------------------------------
    if isinstance(trajs, torch.Tensor):
        trajs = trajs.detach().cpu().numpy()
    trajs = np.asarray(trajs)
    N, T, D = trajs.shape
    if D != 2:
        raise ValueError(f"Expected last dim == 2 (x,y) but got {D}")

    # ---------- convert actions → positions -------------------------------------
    if as_displacement:
        trajs = trajs.cumsum(axis=1)            # integrate displacements

    # ---------- plotting --------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    colormap = plt.cm.get_cmap(cmap, T)

    for n in range(N):
        # Draw grey backbone for the trajectory
        ax.plot(trajs[n, :, 0], trajs[n, :, 1],
                color="grey", alpha=line_alpha, linewidth=1.2)
        # Dots coloured by timestep
        for t in range(T):
            ax.scatter(trajs[n, t, 0], trajs[n, t, 1],
                       s=dot_size,
                       color=colormap(t),
                       edgecolors="k",
                       linewidths=0.25)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Sampled Trajectories (N={N}, T={T})")

    # Colour-bar for timestep reference
    sm = plt.cm.ScalarMappable(cmap=colormap,
                               norm=plt.Normalize(vmin=0, vmax=T - 1))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("timestep")

    plt.show()
