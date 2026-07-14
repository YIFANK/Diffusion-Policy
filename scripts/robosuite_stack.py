"""StackThree: robosuite stacking with THREE color-randomized cubes and a
pair-configurable success check, plus a scripted stacking expert.

The object-selection design mirrors the push two-object benchmark:
  - every scene has 3 cubes with distinct colors (random color->slot binding)
  - the 22-d state is [eef(3), grip(1), (pos(3)+rgb(3)) x 3 cubes] so color is
    IN the observation and "the red block" is groundable from state
  - counterfactual pairing: the same seed (same layout, same colors) is
    demonstrated for ALL 6 ordered (src, tgt) pairs

Usage (expert smoke):
    MUJOCO_GL=egl python robosuite_stack.py --episodes 5
"""
import argparse
import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.stack import Stack
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import (UniformRandomSampler,
                                                SequentialCompositeSampler)

# AGENT-style constrained layout: each cube slot confined to its own y-band
# (colors still bind to slots randomly, so color grounding is untouched —
# only the spatial layout manifold collapses)
ZONE_Y = [(-0.16, -0.08), (-0.04, 0.04), (0.08, 0.16)]

COLOR_RGB = {
    'red':    (0.80, 0.10, 0.10),
    'green':  (0.10, 0.70, 0.10),
    'blue':   (0.10, 0.20, 0.80),
    'yellow': (0.90, 0.80, 0.10),
    'purple': (0.60, 0.10, 0.70),   # held-out color
}
TRAIN_COLORS = ['red', 'green', 'blue', 'yellow']
HELD_COLOR = 'purple'

# Rich palette: makes the COLOR VALUE dimension a covered manifold rather
# than 4 prototypes (the value-manifold analogue of the layout lesson).
RICH_TRAIN_COLORS = {
    'red': (0.80, 0.10, 0.10), 'green': (0.10, 0.70, 0.10),
    'blue': (0.10, 0.20, 0.80), 'yellow': (0.90, 0.80, 0.10),
    'orange': (0.95, 0.55, 0.10), 'pink': (0.95, 0.60, 0.70),
    'brown': (0.55, 0.30, 0.10), 'gray': (0.50, 0.50, 0.50),
    'white': (0.95, 0.95, 0.95), 'black': (0.05, 0.05, 0.05),
    'cyan': (0.10, 0.85, 0.85), 'magenta': (0.90, 0.10, 0.90),
    'lime': (0.55, 0.95, 0.10), 'teal': (0.10, 0.50, 0.50),
    'navy': (0.05, 0.05, 0.45), 'maroon': (0.50, 0.05, 0.10),
    'olive': (0.50, 0.50, 0.10), 'silver': (0.75, 0.75, 0.78),
    'gold': (0.85, 0.70, 0.15), 'beige': (0.90, 0.85, 0.70),
}
RICH_HELD_COLORS = {
    'purple': (0.60, 0.10, 0.70), 'violet': (0.55, 0.35, 0.90),
    'turquoise': (0.25, 0.90, 0.80), 'crimson': (0.85, 0.10, 0.25),
}
COLOR_RGB.update(RICH_TRAIN_COLORS)
COLOR_RGB.update(RICH_HELD_COLORS)

# v5rich2: 10-color palette — keeps value diversity 2.5x the 4-color host while
# keeping the color-TRIPLE space (C(10,3)=120) coverable by ~400 scenes
RICH10 = {k: RICH_TRAIN_COLORS[k] for k in
          ['red', 'green', 'blue', 'yellow', 'orange', 'pink',
           'brown', 'gray', 'cyan', 'gold']}

CUBE_HALF = 0.025  # 5 cm cubes, all equal so any-onto-any stacks


def desc_stack(c_src, c_tgt):
    return f'stack the {c_src} block onto the {c_tgt} block'


class StackThree(Stack):
    """Three equal cubes, colors set per-instance, success = commanded pair."""

    def __init__(self, cube_colors=('red', 'green', 'blue'), constrained=False,
                 **kwargs):
        self.cube_colors = list(cube_colors)
        self.constrained = constrained
        self._target_pair = (0, 1)
        super().__init__(**kwargs)

    def set_target_pair(self, src_idx, tgt_idx):
        self._target_pair = (src_idx, tgt_idx)

    def _load_model(self):
        # bypass Stack._load_model (2 cubes) but keep ManipulationEnv setup
        super(Stack, self)._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        mujoco_arena = TableArena(table_full_size=self.table_full_size,
                                  table_friction=self.table_friction,
                                  table_offset=self.table_offset)
        mujoco_arena.set_origin([0, 0, 0])
        self.cubes = [
            BoxObject(name=f'cube{i}',
                      size_min=[CUBE_HALF] * 3, size_max=[CUBE_HALF] * 3,
                      rgba=list(COLOR_RGB[c]) + [1.0])
            for i, c in enumerate(self.cube_colors)]
        if self.constrained:
            self.placement_initializer = SequentialCompositeSampler(
                name="ObjectSampler")
            for i, cube in enumerate(self.cubes):
                self.placement_initializer.append_sampler(UniformRandomSampler(
                    name=f"zone{i}", mujoco_objects=[cube],
                    x_range=[-0.10, 0.10], y_range=list(ZONE_Y[i]),
                    rotation=None, ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset, z_offset=0.01,
                    rng=self.rng))
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler", mujoco_objects=self.cubes,
                x_range=[-0.10, 0.10], y_range=[-0.16, 0.16], rotation=None,
                ensure_object_boundary_in_range=False, ensure_valid_placement=True,
                reference_pos=self.table_offset, z_offset=0.01, rng=self.rng)
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cubes)

    def _setup_references(self):
        super(Stack, self)._setup_references()
        self.cube_body_ids = [self.sim.model.body_name2id(c.root_body)
                              for c in self.cubes]

    def _setup_observables(self):
        # skip Stack's cubeA/cubeB sensors; we read positions from sim directly
        return super(Stack, self)._setup_observables()

    def cube_pos(self, i):
        return np.array(self.sim.data.body_xpos[self.cube_body_ids[i]])

    def reward(self, action=None):
        return float(self._check_success())

    def _check_success(self):
        si, ti = self._target_pair
        src, tgt = self.cube_pos(si), self.cube_pos(ti)
        horiz = np.linalg.norm(src[:2] - tgt[:2]) < 0.03
        dz = src[2] - tgt[2]
        stacked = 2 * CUBE_HALF - 0.015 < dz < 2 * CUBE_HALF + 0.015
        touching = self.check_contact(self.cubes[si], self.cubes[ti])
        grasping = self._check_grasp(gripper=self.robots[0].gripper,
                                     object_geoms=self.cubes[si])
        return bool(horiz and stacked and touching and not grasping)


def make_stack_env(seed, colors, constrained=False, cam=96):
    env = suite.make('StackThree', robots='Panda', cube_colors=colors,
                     constrained=constrained,
                     has_renderer=False, has_offscreen_renderer=True,
                     use_camera_obs=True, camera_names='agentview',
                     camera_heights=cam, camera_widths=cam, control_freq=20,
                     reward_shaping=False, seed=seed)
    return env


def rel22(s):
    """Object-centric transform of the 22-d state: cube positions become
    eef-relative offsets (grasping/placing is translation-invariant, which
    collapses the 6-d layout manifold the absolute encoding must cover).
    Works on (..., 22) arrays; eef, gripper and RGB dims unchanged."""
    s = np.array(s, dtype=np.float32, copy=True)
    for i in range(3):
        s[..., 4 + 6 * i:7 + 6 * i] -= s[..., 0:3]
    return s


def state22(obs, env):
    parts = [obs['robot0_eef_pos'], [obs['robot0_gripper_qpos'][0]]]
    for i, c in enumerate(env.cube_colors):
        parts.append(env.cube_pos(i))
        parts.append(COLOR_RGB[c])
    return np.concatenate([np.asarray(p, dtype=np.float32).ravel()
                           for p in parts])


class StackExpert:
    """Waypoint state machine: above-src -> center -> descend -> grasp ->
    lift -> above-tgt -> lower -> release -> retreat."""

    def __init__(self, env, src_idx, tgt_idx, kp=6.0, max_step=0.9):
        self.env, self.si, self.ti = env, src_idx, tgt_idx
        self.kp, self.max_step = kp, max_step
        self.phase = 0
        self.grip_count = 0

    def act(self, obs):
        eef = obs['robot0_eef_pos']
        src = self.env.cube_pos(self.si)
        tgt = self.env.cube_pos(self.ti)
        grip = -1.0
        target = eef
        LIFT = 1.02
        if self.phase == 0:      # above source cube
            target = src + np.array([0, 0, 0.10])
            if np.linalg.norm(eef[:2] - src[:2]) < 0.012 and eef[2] - src[2] > 0.08:
                self.phase = 1
        elif self.phase == 1:    # descend
            target = src + np.array([0, 0, 0.001])
            if eef[2] - src[2] < 0.008:
                self.phase = 2
        elif self.phase == 2:    # close gripper
            target = eef
            grip = 1.0
            self.grip_count += 1
            if self.grip_count > 8:
                self.phase = 3
        elif self.phase == 3:    # lift
            grip = 1.0
            target = np.array([eef[0], eef[1], LIFT])
            if eef[2] > LIFT - 0.02:
                self.phase = 4
        elif self.phase == 4:    # transport above target
            grip = 1.0
            target = np.array([tgt[0], tgt[1], LIFT])
            if np.linalg.norm(eef[:2] - tgt[:2]) < 0.012:
                self.phase = 5
        elif self.phase == 5:    # lower onto target
            grip = 1.0
            target = np.array([tgt[0], tgt[1], tgt[2] + 2 * CUBE_HALF + 0.015])
            if eef[2] - tgt[2] < 2 * CUBE_HALF + 0.02:
                self.phase = 6
        else:                    # release + retreat
            grip = -1.0
            target = np.array([eef[0], eef[1], LIFT])
        d = np.clip(self.kp * (target - eef), -self.max_step, self.max_step)
        return np.concatenate([d, np.zeros(3), [grip]])


def rollout_expert(seed, colors, src_idx, tgt_idx, max_steps=300, record=False,
                   constrained=False):
    env = make_stack_env(seed, colors, constrained=constrained)
    obs = env.reset()
    env.set_target_pair(src_idx, tgt_idx)
    ex = StackExpert(env, src_idx, tgt_idx)
    frames, states, actions = [], [], []
    for t in range(max_steps):
        a = ex.act(obs)
        if record:
            frames.append(obs['agentview_image'][::-1].copy())
            states.append(state22(obs, env))
            actions.append(a.copy())
        obs, r, done, info = env.step(a)
        if env._check_success():
            env.close()
            return True, t, frames, states, actions
    env.close()
    return False, max_steps, frames, states, actions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    ok, lens = 0, []
    for i in range(args.episodes):
        colors = list(rng.choice(TRAIN_COLORS, size=3, replace=False))
        si, ti = rng.choice(3, size=2, replace=False)
        s, t, *_ = rollout_expert(100 + i, colors, int(si), int(ti))
        ok += s
        lens.append(t)
        print(f'ep{i} {colors[si]}->{colors[ti]}: success={s} steps={t}', flush=True)
    print(f'STACKEXPERT: {ok}/{args.episodes} mean_steps={np.mean(lens):.0f}',
          flush=True)
