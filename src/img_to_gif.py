from PIL import Image
import os
from typing import Sequence, Union
import numpy as np

def np_to_img(arr: np.ndarray) -> Image.Image:
    """
    Convert a NumPy array to a PIL.Image.Image.

    Parameters:
    - arr: numpy.ndarray
        Should be either grayscale (H, W) or color (H, W, 3) or (H, W, 4).

    Returns:
    - PIL.Image.Image
    """
    if arr.dtype != np.uint8:
        arr = (255 * np.clip(arr, 0, 1)).astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr, mode='L')
    elif arr.ndim == 3:
        if arr.shape[2] == 3:
            arr = arr[..., ::-1]  # BGR → RGB
            return Image.fromarray(arr, mode='RGB')
        else:
            raise ValueError(f"Unsupported channel size: {arr.shape[2]}")
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

def images_to_gif(
    frames: Sequence[Union[str, Image.Image]],
    save_path: str,
    fps: int = 10,  # Frames per second
    loop: int = 0          # 0 = loop forever, 1 = play once, etc.
) -> None:
    """
    Convert a sequence of images to a GIF.

    Parameters
    ----------
    frames : list[str | PIL.Image.Image]
        Either file paths or already-loaded PIL Images, in the order you want them to appear.
    save_path : str
        Where to write the resulting GIF (e.g. 'outputs/animation.gif').
    duration : int, default 100
        Frame delay in milliseconds.
    loop : int, default 0
        How many times to repeat the GIF (0 = infinite).
    """
    if not frames:
        raise ValueError("frames cannot be empty")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Load any paths into PIL.Image.Image objects
    pil_frames = []
    for f in frames:
        if isinstance(f, str):
            if not os.path.exists(f):
                raise FileNotFoundError(f"File not found: {f}")
            img = Image.open(f)
        elif isinstance(f, Image.Image):
            img = f
        elif isinstance(f, np.ndarray):
            img = np_to_img(f)
        pil_frames.append(img)
    duration = int(1000 / fps)  # Convert FPS to duration in milliseconds
    # Save: first frame plus the rest in append_images
    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:],
        optimize=True,
        loop=loop,
        duration=duration,
    )
