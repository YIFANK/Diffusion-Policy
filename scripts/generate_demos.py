"""Scripted expert demo generator for the push workspace.

Regenerates the (gitignored, lost) `dataset/{Color}_{num}.pkl` demonstration
files using a simple servoing push policy instead of human teleop:
at every control step the agent re-computes the block->corner direction,
navigates around the block to get behind it, then pushes through the
block center toward the target corner.

Data is saved in the same Scene/Trajectory/State pickle format that
`tiny_embodied_reasoning.workspace.utils` uses, so
`diffusion_policy.data.dataset.preprocess` loads it unchanged.

Usage:
    python generate_demos.py --colors Blue --episodes 30
    python generate_demos.py --colors Blue Red Green --episodes 30
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import copy
import numpy as np
from omegaconf import OmegaConf
from tiny_embodied_reasoning.environment import env as ter_env
from tiny_embodied_reasoning.observers.observer import StateObserver, ImageObserver
from tiny_embodied_reasoning.workspace.utils import State, Trajectory, Scene, save_trajectory_pickle

ENV_YAML = """
info:
    render_mode: 'rgb_array'
    video_fps: 10
    sim_hz: 100
    control_hz: 10
    render_size: 96
    window_height: 512
    window_width: 512
    seed: 0
scene_info:
    o1:
        geometry: {{shape: 'box', width: 100, height: 100}}
        color: '{color}'
        angle: 0
agent_info:
    observer: {{type: 'image', verbose: False}}
"""

# corner index convention matches scripts/inference.py win_condition
CORNER_NAMES = ['lower-right', 'upper-right', 'upper-left', 'lower-left']
CORNER_TARGETS = {
    0: np.array([462.0, 50.0]),   # lower-right: x>416, y<96
    1: np.array([462.0, 462.0]),  # upper-right: x>416, y>416
    2: np.array([50.0, 462.0]),   # upper-left:  x<96,  y>416
    3: np.array([50.0, 50.0]),    # lower-left:  x<96,  y<96
}
WIN_CONDITION = [
    lambda p: p[0] > 416 and p[1] < 96,
    lambda p: p[0] > 416 and p[1] > 416,
    lambda p: p[0] < 96 and p[1] > 416,
    lambda p: p[0] < 96 and p[1] < 96,
]

BLOCK_HALF = 50.0
AGENT_R = 15.0
BEHIND_OFFSET = BLOCK_HALF + AGENT_R + 30.0   # standoff point behind the block
AVOID_R = BLOCK_HALF + AGENT_R + 20.0         # clearance radius when circling around
MAX_CMD_STEP = 35.0                           # px commanded per control step


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros_like(v)


def push_direction(block, goal):
    """Axis-aligned push direction: work on the axis with the larger error.

    Axis-aligned pushes are far more reliable than diagonal ones for a
    square block (no rotation-induced slip, no fighting the wall when
    finishing the second axis inside the corner region).
    """
    err = goal - block
    axis = int(abs(err[1]) > abs(err[0]))     # 0 = x, 1 = y
    # if the dominant axis is nearly done, switch to the other one
    if abs(err[axis]) < 20.0 and abs(err[1 - axis]) >= 20.0:
        axis = 1 - axis
    d = np.zeros(2)
    d[axis] = np.sign(err[axis]) if err[axis] != 0 else 1.0
    return d


def scripted_push_action(agent, block, goal, side=None, max_cmd_step=MAX_CMD_STEP):
    """One servoing step. Returns the commanded target position (2,).

    Behavior-mode factors (see proposal §6.1):
      side: None = shorter arc; +1 / -1 force counterclockwise / clockwise
            navigation around the block ("route" factor).
      max_cmd_step: commanded px per control step ("force" factor —
            gentle vs aggressive push speed).
    """
    d = push_direction(block, goal)           # push direction
    behind = block - d * BEHIND_OFFSET        # where we want to stand
    to_block = agent - block

    # Are we behind the block (within ~40 deg of the -d axis) and close?
    aligned = np.dot(unit(to_block), -d) > 0.75
    close = np.linalg.norm(to_block) < BEHIND_OFFSET + 25.0

    if aligned and close:
        # push through the block center toward the goal
        target = block + d * 10.0
    else:
        # navigate to the standoff point by walking along a circle around
        # the block until we are behind it (never cuts through the block)
        ang_agent = np.arctan2(to_block[1], to_block[0])
        ang_behind = np.arctan2(-d[1], -d[0])
        delta = (ang_behind - ang_agent + np.pi) % (2 * np.pi) - np.pi
        if side is not None and abs(delta) > 0.35:
            # force the requested rotation direction (take the long way
            # around if needed)
            delta = delta if np.sign(delta) == side else delta + side * 2 * np.pi
        if abs(delta) < 0.35:
            target = behind
        else:
            ang_next = ang_agent + np.sign(delta) * 0.45
            target = block + np.array([np.cos(ang_next), np.sin(ang_next)]) * (AVOID_R + 10.0)
    # keep commands inside the arena walls
    target = np.clip(target, 25.0, 487.0)

    # cap the commanded displacement so the PD controller stays smooth
    step = target - agent
    dist = np.linalg.norm(step)
    if dist > max_cmd_step:
        step = step / dist * max_cmd_step
    return agent + step


def rollout(color, corner_idx, seed, max_steps=300, side=None, max_cmd_step=MAX_CMD_STEP):
    """Run one scripted episode. Returns (Trajectory, success)."""
    cfg = OmegaConf.create(ENV_YAML.format(color=color))
    env = ter_env.TEREnv(**cfg.info, scene_info=cfg.scene_info,
                         agent_info=cfg.agent_info, verbose=False)
    env.set_seed(seed)
    env._setup()
    image_observer = ImageObserver(env, render_size=96, verbose=False)
    state_observer = StateObserver(env, verbose=False)
    goal = CORNER_TARGETS[corner_idx]

    traj = Trajectory(seed=seed, data=[])
    success = False
    for _ in range(max_steps):
        img = image_observer.observe()
        state = copy.deepcopy(state_observer.observe())
        agent = np.array(env.agent.position, dtype=np.float64)
        block = np.array(state['o1']['position'], dtype=np.float64)

        action = scripted_push_action(agent, block, goal, side=side,
                                      max_cmd_step=max_cmd_step)
        traj.data.append(State(state=state, action=tuple(action), observation=img.copy()))
        env.step(action)

        new_block = np.array(state_observer.observe()['o1']['position'], dtype=np.float64)
        if WIN_CONDITION[corner_idx](new_block):
            success = True
            break
    return traj, success


TWO_OBJ_YAML = """
info:
    render_mode: 'rgb_array'
    video_fps: 10
    sim_hz: 100
    control_hz: 10
    render_size: 96
    window_height: 512
    window_width: 512
    seed: 0
scene_info:
    o1:
        geometry: {o1_geom}
        color: '{o1_color}'
        angle: 0
    o2:
        geometry: {o2_geom}
        color: '{o2_color}'
        angle: 0
agent_info:
    observer: {{type: 'image', verbose: False}}
"""
BOX_GEOM = "{shape: 'box', width: 100, height: 100}"
CIRCLE_GEOM = "{shape: 'circle', radius: 55}"

# two-object scene configs: (o1_geom, o1_color, o2_geom, o2_color)
# color pair: identity differs by color; shape pair: identity differs by shape
TWO_OBJ_SCENES = {
    'color': (BOX_GEOM, 'Blue', BOX_GEOM, 'Red'),
    'shape': (BOX_GEOM, 'Green', CIRCLE_GEOM, 'Green'),
}
# target name -> (scene, object key)
TWO_OBJ_TARGETS = {
    'blue':   ('color', 'o1'),
    'red':    ('color', 'o2'),
    'square': ('shape', 'o1'),
    'circle': ('shape', 'o2'),
}
TWO_OBJ_GOAL_CORNER = 0  # fixed goal: lower-right


def rollout_two_obj(target_name, seed, max_steps=350):
    """One scripted episode in a two-object scene: push the named object to the
    fixed goal corner; the other object is a distractor."""
    scene_key, target_key = TWO_OBJ_TARGETS[target_name]
    g1, c1, g2, c2 = TWO_OBJ_SCENES[scene_key]
    cfg = OmegaConf.create(TWO_OBJ_YAML.format(o1_geom=g1, o1_color=c1,
                                               o2_geom=g2, o2_color=c2))
    env = ter_env.TEREnv(**cfg.info, scene_info=cfg.scene_info,
                         agent_info=cfg.agent_info, verbose=False)
    env.set_seed(seed)
    env._setup()
    image_observer = ImageObserver(env, render_size=96, verbose=False)
    state_observer = StateObserver(env, verbose=False)
    goal = CORNER_TARGETS[TWO_OBJ_GOAL_CORNER]
    distractor_key = 'o2' if target_key == 'o1' else 'o1'

    traj = Trajectory(seed=seed, data=[])
    success = False
    for _ in range(max_steps):
        img = image_observer.observe()
        state = copy.deepcopy(state_observer.observe())
        agent = np.array(env.agent.position, dtype=np.float64)
        block = np.array(state[target_key]['position'], dtype=np.float64)
        distractor = np.array(state[distractor_key]['position'], dtype=np.float64)

        action = scripted_push_action(agent, block, goal)
        # if the commanded point sits inside the distractor's footprint,
        # deflect it away (cheap obstacle avoidance; retries cover the rest)
        away = action - distractor
        d = np.linalg.norm(away)
        if d < AVOID_R:
            action = distractor + away / max(d, 1e-6) * AVOID_R
        traj.data.append(State(state=state, action=tuple(action), observation=img.copy()))
        env.step(action)

        new_block = np.array(state_observer.observe()[target_key]['position'], dtype=np.float64)
        if WIN_CONDITION[TWO_OBJ_GOAL_CORNER](new_block):
            success = True
            break
    return traj, success


def generate_two_obj_paired(scene_key, n_pairs, out_dir, seed0=0):
    """Strict counterfactual pairs: for each seed (= layout), roll out BOTH
    targets of the scene; keep the pair only if both succeed. Every kept
    layout therefore appears with both instructions — vision alone cannot
    fit the data."""
    targets = [t for t, (sk, _) in TWO_OBJ_TARGETS.items() if sk == scene_key]
    assert len(targets) == 2
    scenes = {t: Scene(config={'target': t, 'goal': CORNER_NAMES[TWO_OBJ_GOAL_CORNER],
                               'generator': 'scripted_push_two_obj_paired_v1',
                               'paired_with': [x for x in targets if x != t][0]},
                       trajectories=[]) for t in targets}
    seed = seed0
    attempts = 0
    while len(scenes[targets[0]].trajectories) < n_pairs and attempts < n_pairs * 6:
        rolls = {t: rollout_two_obj(t, seed) for t in targets}
        seed += 1
        attempts += 1
        if all(success for _, success in rolls.values()):
            for t in targets:
                scenes[t].trajectories.append(rolls[t][0])
    n = len(scenes[targets[0]].trajectories)
    print(f"paired scene '{scene_key}' ({'+'.join(targets)}): "
          f"{n}/{n_pairs} pairs in {attempts} attempts", flush=True)
    for t in targets:
        out_path = os.path.join(out_dir, f'TwoObj_{t}.pkl')
        if os.path.exists(out_path):
            os.remove(out_path)
        save_trajectory_pickle(out_path, scenes[t])
    return n


def generate_two_obj(target_name, n_episodes, out_path, seed0=0):
    scene = Scene(config={'target': target_name, 'goal': CORNER_NAMES[TWO_OBJ_GOAL_CORNER],
                          'generator': 'scripted_push_two_obj_v1'}, trajectories=[])
    seed = seed0
    attempts = 0
    while len(scene.trajectories) < n_episodes and attempts < n_episodes * 5:
        traj, success = rollout_two_obj(target_name, seed)
        seed += 1
        attempts += 1
        if success:
            scene.trajectories.append(traj)
    print(f"two-obj target={target_name}: "
          f"{len(scene.trajectories)}/{n_episodes} successes in {attempts} attempts")
    if os.path.exists(out_path):
        os.remove(out_path)
    save_trajectory_pickle(out_path, scene)


SPEED_MODES = {'gentle': 20.0, 'fast': 40.0}
SIDE_MODES = {'ccw': 1, 'cw': -1}


def generate(color, corner_idx, n_episodes, out_path, seed0=0,
             side_name=None, speed_name=None):
    side = SIDE_MODES[side_name] if side_name else None
    max_cmd = SPEED_MODES[speed_name] if speed_name else MAX_CMD_STEP
    max_steps = 700 if speed_name == 'gentle' else 300
    scene = Scene(config={'color': color, 'corner': CORNER_NAMES[corner_idx],
                          'side': side_name, 'speed': speed_name,
                          'generator': 'scripted_push_v2'}, trajectories=[])
    seed = seed0
    attempts = 0
    while len(scene.trajectories) < n_episodes and attempts < n_episodes * 4:
        traj, success = rollout(color, corner_idx, seed, max_steps=max_steps,
                                side=side, max_cmd_step=max_cmd)
        seed += 1
        attempts += 1
        if success:
            scene.trajectories.append(traj)
    mode_str = f" [{side_name or 'auto'}/{speed_name or 'normal'}]"
    print(f"{color} -> {CORNER_NAMES[corner_idx]}{mode_str}: "
          f"{len(scene.trajectories)}/{n_episodes} successes in {attempts} attempts")
    if os.path.exists(out_path):
        os.remove(out_path)
    save_trajectory_pickle(out_path, scene)
    return len(scene.trajectories)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--colors', nargs='+', default=['Blue'])
    parser.add_argument('--corners', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--out_dir', default='../dataset')
    parser.add_argument('--modes', action='store_true',
                        help='generate the factored behavior-mode datasets '
                             '(side x speed) instead of the plain ones')
    parser.add_argument('--two_objects', action='store_true',
                        help='generate the two-object object-selection datasets')
    parser.add_argument('--paired', action='store_true',
                        help='two-object: same seeds for targets in the same scene '
                             '(literal same-layout counterfactual pairs)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.two_objects:
        if args.paired:
            # strict same-layout counterfactual pairs, both targets per seed
            for scene_key in TWO_OBJ_SCENES:
                generate_two_obj_paired(scene_key, args.episodes, args.out_dir,
                                        seed0=hash(('twoobj', scene_key)) % 100000)
        else:
            for target in TWO_OBJ_TARGETS:
                out = os.path.join(args.out_dir, f'TwoObj_{target}.pkl')
                generate_two_obj(target, args.episodes, out,
                                 seed0=hash(('twoobj', target)) % 100000)
        sys.exit(0)
    for color in args.colors:
        for corner in args.corners:
            if args.modes:
                for side_name in SIDE_MODES:
                    for speed_name in SPEED_MODES:
                        out = os.path.join(
                            args.out_dir,
                            f'{color}_{corner}_{side_name}_{speed_name}.pkl')
                        generate(color, corner, args.episodes, out,
                                 seed0=hash((color, corner, side_name, speed_name)) % 100000,
                                 side_name=side_name, speed_name=speed_name)
            else:
                out = os.path.join(args.out_dir, f'{color}_{corner}.pkl')
                generate(color, corner, args.episodes, out,
                         seed0=hash((color, corner)) % 100000)
