"""Train the v2 behavior-mode-conditioned diffusion policy (proposal Variant A).

Data: dataset_modes/Blue_{corner}_{side}_{speed}.pkl (4 corners x cw/ccw x gentle/fast).
Conditioning: composable descriptions
    "push the blue block to the {corner} corner, approaching {side}, {speed}".

HELD-OUT COMPOSITION SPLIT (E4): two (corner, side, speed) combos are excluded
from training. At test time we infer z_route and z_force from *other* combos'
demos and compose them to solve the held-out ones.

Usage:
    python train_modes.py --epochs 100
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from train import train_diffusion_policy
from cache_text_embeddings import SIDE_PHRASES, SPEED_PHRASES

CORNER_NAMES = ['lower-right', 'upper-right', 'upper-left', 'lower-left']

# held out for the E4 composition test — never seen in training
HELD_OUT = [
    (2, 'cw', 'gentle'),   # upper-left, clockwise, gentle
    (0, 'ccw', 'fast'),    # lower-right, counterclockwise, fast
]


def build_split(color='Blue'):
    train_paths, train_descs, held_paths, held_descs = [], [], [], []
    for corner in range(4):
        for side in ['cw', 'ccw']:
            for speed in ['gentle', 'fast']:
                path = f'../dataset_modes/{color}_{corner}_{side}_{speed}.pkl'
                desc = (f'push the {color.lower()} block to the {CORNER_NAMES[corner]} corner, '
                        f'{SIDE_PHRASES[side]}, {SPEED_PHRASES[speed]}')
                if (corner, side, speed) in HELD_OUT:
                    held_paths.append(path)
                    held_descs.append(desc)
                else:
                    train_paths.append(path)
                    train_descs.append(desc)
    return train_paths, train_descs, held_paths, held_descs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--model_path', default='../trained_models/dp_unet_clip_modes_v2.pth')
    parser.add_argument('--data_dir', default='../dataset_modes',
                        help='mode dataset directory (e.g. ../dataset_modes_60 for the 4x set)')
    parser.add_argument('--lowdim', action='store_true',
                        help='state-based policy: obs = agent+block positions, no vision')
    args = parser.parse_args()

    train_paths, train_descs, held_paths, held_descs = build_split()
    train_paths = [p.replace('../dataset_modes/', args.data_dir.rstrip('/') + '/')
                   for p in train_paths]
    print(f"training on {len(train_paths)} combos from {args.data_dir}; held out: {held_descs}")
    train_diffusion_policy(epochs=args.epochs, logging=args.logging,
                           noise_pred_net_type='unet', model_path=args.model_path,
                           dataset_paths=train_paths, descriptions=train_descs,
                           vision=not args.lowdim,
                           state_keys=('agent', 'o1') if args.lowdim else ('agent',))
