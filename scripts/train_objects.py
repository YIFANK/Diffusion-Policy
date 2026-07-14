"""Train the v4 object-selection policy on two-object scenes.

Data: dataset_twoobj/TwoObj_{blue,red,square,circle}.pkl — two objects per
scene, push the named one to the fixed lower-right corner. Tests whether text
conditioning can select WHICH object to act on (selection factor), given that
training scenes contain multiple objects.

Control experiment for the 2025 observation that single-object-trained policies
cannot do object selection zero-shot: evaluate v1 on these scenes and compare.

Usage:
    python train_objects.py --epochs 100
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from train import train_diffusion_policy

TARGETS = ['blue', 'red', 'square', 'circle']

def build_lists(data_dir='../dataset_twoobj'):
    paths = [f'{data_dir}/TwoObj_{t}.pkl' for t in TARGETS]
    descs = [f'push the {t} block to the lower-right corner' for t in TARGETS]
    return paths, descs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--model_path', default='../trained_models/dp_unet_clip_twoobj_v4.pth')
    parser.add_argument('--data_dir', default='../dataset_twoobj')
    parser.add_argument('--p_uncond', type=float, default=0.1)
    parser.add_argument('--early_frames', type=int, default=0,
                        help='upweight the first N frames of each episode')
    parser.add_argument('--early_weight', type=float, default=5.0)
    parser.add_argument('--text_lr', type=float, default=1e-2)
    parser.add_argument('--text_gate', action='store_true')
    parser.add_argument('--lowdim', action='store_true',
                        help='state policy: obs = agent+o1+o2 positions (6-d)')
    args = parser.parse_args()

    paths, descs = build_lists(args.data_dir)
    train_diffusion_policy(epochs=args.epochs, logging=args.logging,
                           noise_pred_net_type='unet', model_path=args.model_path,
                           dataset_paths=paths, descriptions=descs,
                           p_uncond=args.p_uncond,
                           early_frames=args.early_frames,
                           early_weight=args.early_weight,
                           text_lr=args.text_lr, text_gate=args.text_gate,
                           vision=not args.lowdim,
                           state_keys=('agent', 'o1', 'o2') if args.lowdim else ('agent',))
