"""E2: few-shot concept inversion on the frozen v1 policy.

For each task: freeze the trained policy, infer a concept embedding from K
demonstrations by optimizing the score-space composition loss
(concept_inference.infer_new_concepts), then evaluate the inferred concept
closed-loop. Sweeps K to measure few-shot scaling (proposal §7.5).

Usage:
    python eval_e2.py --model_path ../trained_models/dp_unet_clip_scripted_v1.pth \
                      --colors blue --K 1 4 --episodes 10
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
from concept_inference import infer_new_concepts
from inference import evaluate

COLORS_CAP = {'blue': 'Blue', 'red': 'Red', 'green': 'Green'}
CORNER_NAMES = ['lower-right', 'upper-right', 'upper-left', 'lower-left']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/dp_unet_clip_scripted_v1.pth')
    parser.add_argument('--colors', nargs='+', default=['blue'])
    parser.add_argument('--K', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--inversion_epochs', type=int, default=300)
    parser.add_argument('--init_type', default='rand')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    results = {}
    for color in args.colors:
        for num in range(4):
            demo_path = f'../dataset/{COLORS_CAP[color]}_{num}.pkl'
            for k_shot in args.K:
                tag = f'{color}_{CORNER_NAMES[num]}_K{k_shot}'
                print(f'=== E2 inversion: {tag} ===', flush=True)
                learned = infer_new_concepts(
                    model_path=args.model_path, policy_type='diffusion',
                    new_concept_dataset_path=demo_path,
                    weights_output_path=f'../output/concepts/e2_{tag}_weights.npy',
                    embeddings_output_path=f'../output/concepts/e2_{tag}_embeddings.npy',
                    num_epochs=args.inversion_epochs, learning_rate=1e-2,
                    K=1, logging=False, init_type=args.init_type,
                    task=[color, num], num_trajectories=k_shot)
                score = evaluate(
                    max_steps=200, num_episodes=args.episodes,
                    model_path=args.model_path, render=False,
                    policy_type='diffusion', task=[color, num],
                    concept_weights=learned['weights'],
                    concept_embeddings=learned['embeddings'],
                    init_type=f'e2_{args.init_type}_K{k_shot}')
                results[tag] = score
                print(f'E2_RESULT {tag}: {score}/{args.episodes}', flush=True)

    out = args.out or args.model_path.replace('.pth', '_e2_eval.json')
    with open(out, 'w') as f:
        json.dump({'model': os.path.basename(args.model_path),
                   'episodes_per_task': args.episodes,
                   'init_type': args.init_type,
                   'inversion_epochs': args.inversion_epochs,
                   'per_task': results}, f, indent=2)
    print('E2_DONE', json.dumps(results), flush=True)
