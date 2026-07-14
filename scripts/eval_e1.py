"""E1 baseline evaluation: closed-loop success rate of the goal-conditioned
policy on all 12 (color x corner) tasks with original text embeddings.

Writes a JSON summary next to the model plus the per-episode logs that
`inference.evaluate` already produces under output/eval/<model_name>/.

Usage:
    python eval_e1.py --model_path ../trained_models/dp_unet_clip_scripted_v1.pth \
                      --episodes 10 [--render]
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
from inference import evaluate

COLORS = ['blue', 'red', 'green']
CORNER_NAMES = ['lower-right', 'upper-right', 'upper-left', 'lower-left']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/dp_unet_clip_scripted_v1.pth')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--colors', nargs='+', default=COLORS)
    parser.add_argument('--lowdim', action='store_true',
                        help='evaluate a state-based policy (obs = agent+block positions)')
    args = parser.parse_args()
    vision = not args.lowdim
    state_keys = ('agent', 'o1') if args.lowdim else ('agent',)

    results = {}
    total = 0
    for color in args.colors:
        for num in range(4):
            task_name = f'{color}_{CORNER_NAMES[num]}'
            print(f'=== evaluating {task_name} ===', flush=True)
            score = evaluate(max_steps=args.max_steps, num_episodes=args.episodes,
                             model_path=args.model_path, render=args.render,
                             policy_type='diffusion', task=[color, num],
                             vision=vision, state_keys=state_keys)
            results[task_name] = score
            total += score
            print(f'EVAL_RESULT {task_name}: {score}/{args.episodes}', flush=True)

    n_tasks = len(args.colors) * 4
    summary = {
        'model': os.path.basename(args.model_path),
        'episodes_per_task': args.episodes,
        'per_task': results,
        'total': total,
        'max_total': n_tasks * args.episodes,
        'mean_success_rate': total / (n_tasks * args.episodes),
    }
    out = args.model_path.replace('.pth', '_e1_eval.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print('EVAL_DONE', json.dumps(summary['per_task']))
    print(f"TOTAL {total}/{n_tasks * args.episodes} "
          f"(mean {summary['mean_success_rate']:.2f}) -> {out}", flush=True)
