"""Build output/cached_labels.pkl: {task description -> frozen text embedding}.

The policy's trainable text_encoder MLP projects these frozen embeddings to
64-d conditioning vectors, and auto-detects the input dim from this cache
(512 for CLIP, 3584 for VLM2Vec).

Usage:
    python cache_text_embeddings.py            # CLIP (default, light)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import torch

CORNER_NAMES = ['lower-right', 'upper-right', 'upper-left', 'lower-left']
COLORS = ['blue', 'red', 'green']
# base goal-only tasks (v1)
DESCRIPTIONS = [f'push the {c} block to the {corner} corner'
                for c in COLORS for corner in CORNER_NAMES]
# behavior-mode tasks (v2): route (approach side) x force (push speed)
SIDE_PHRASES = {'cw': 'approaching clockwise', 'ccw': 'approaching counterclockwise'}
SPEED_PHRASES = {'gentle': 'gently', 'fast': 'quickly'}
DESCRIPTIONS += [
    f'push the {c} block to the {corner} corner, {SIDE_PHRASES[side]}, {SPEED_PHRASES[speed]}'
    for c in COLORS for corner in CORNER_NAMES
    for side in SIDE_PHRASES for speed in SPEED_PHRASES
]
# two-object object-selection tasks (fixed goal corner = lower-right)
TWO_OBJ_DESCRIPTIONS = [
    'push the blue block to the lower-right corner',      # already in base set
    'push the red block to the lower-right corner',       # already in base set
    'push the square block to the lower-right corner',
    'push the circle block to the lower-right corner',
]
DESCRIPTIONS += [d for d in TWO_OBJ_DESCRIPTIONS if d not in DESCRIPTIONS]
# PushT factored tasks (rotation-augmented goals x replay speed)
DESCRIPTIONS += [
    f'push the T to the goal rotated {deg} degrees, {sp}'
    for deg in ['0', '90', '180', '270'] for sp in ['quickly', 'slowly']
]
# robosuite PickPlaceCan factored tasks
DESCRIPTIONS += [
    f'pick the can approaching from the {s} and place it in the bin, '
    f'{"via a high arc" if a == "high" else "directly"}, {sp}'
    for s in ['left', 'right'] for a in ['direct', 'high']
    for sp in ['quickly', 'slowly']
]
# StackThree color-pair tasks (incl. held-out colors and the rich palette)
STACK_COLORS = ['red', 'green', 'blue', 'yellow', 'purple',
                'orange', 'pink', 'brown', 'gray', 'white', 'black', 'cyan',
                'magenta', 'lime', 'teal', 'navy', 'maroon', 'olive',
                'silver', 'gold', 'beige', 'violet', 'turquoise', 'crimson']
DESCRIPTIONS += [
    f'stack the {c1} block onto the {c2} block'
    for c1 in STACK_COLORS for c2 in STACK_COLORS if c1 != c2
]
OUT_PATH = '../output/cached_labels.pkl'


def encode_clip(texts):
    from transformers import CLIPTokenizer, CLIPTextModel
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    out = {}
    with torch.no_grad():
        for t in texts:
            tokens = tokenizer(text=t, padding=True, return_tensors="pt")
            # CLIP's text transformer is causal: position 0 is the BOS token,
            # which is identical for every input. The sentence embedding is the
            # EOS-pooled output (pooler_output).
            out[t] = model(**tokens).pooler_output  # (1, 512)
    return out


def encode_clip_tokens(texts):
    """Per-token last_hidden_state (real tokens only) for the x-attn ablation.
    Includes the empty string as the CFG unconditional sequence."""
    from transformers import CLIPTokenizer, CLIPTextModel
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    out = {}
    with torch.no_grad():
        for t in list(texts) + ['']:
            tokens = tokenizer(text=t, padding=True, return_tensors="pt")
            hidden = model(**tokens).last_hidden_state[0]      # (L, 512)
            L = int(tokens['attention_mask'][0].sum())
            out[t] = hidden[:L].numpy().astype('float16')
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', action='store_true',
                        help='also build the per-token cache (x-attn ablation)')
    args = parser.parse_args()

    embeddings = encode_clip(DESCRIPTIONS)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(embeddings, f)
    dim = next(iter(embeddings.values())).shape
    print(f"Saved {len(embeddings)} embeddings of shape {dim} to {OUT_PATH}")

    if args.tokens:
        tok = encode_clip_tokens(DESCRIPTIONS)
        with open(OUT_PATH.replace('cached_labels', 'cached_tokens'), 'wb') as f:
            pickle.dump(tok, f)
        lens = [len(v) for v in tok.values()]
        print(f"Saved {len(tok)} token sequences (len {min(lens)}-{max(lens)})")
