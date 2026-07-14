"""Scripted factored expert for robosuite PickPlaceCan (Panda, OSC_POSE).

Factors (v0):
  side:  approach the can via a waypoint offset to its left or right (y-axis)
         before centering — the route factor.
  arc:   transport directly at low height vs via a high arc — the style factor.
  speed: P-gain / step scaling — gentle vs fast.

Waypoint state machine: above-side -> above -> descend -> grasp -> lift ->
transport (arc/direct) -> above-bin -> descend -> release.

Usage:
    MUJOCO_GL=egl python robosuite_expert.py --episodes 5 --side left --arc high --speed fast
"""
import argparse
import numpy as np
import robosuite as suite

def can_target_xy(env):
    """The can's designated quadrant center in bin2 (read from the env)."""
    return env.target_bin_placements[env.object_to_id['can']][:2].copy()


def make_env(seed=None):
    env = suite.make(
        'PickPlaceCan', robots='Panda', has_renderer=False,
        has_offscreen_renderer=True, use_camera_obs=True,
        camera_names='agentview', camera_heights=96, camera_widths=96,
        control_freq=20, reward_shaping=False)
    if seed is not None:
        np.random.seed(seed)
    return env


class Expert:
    def __init__(self, side='left', arc='direct', speed='fast', target_xy=None):
        self.side = 1.0 if side == 'left' else -1.0
        self.arc = arc
        self.kp = 6.0 if speed == 'fast' else 2.5
        self.max_step = 0.9 if speed == 'fast' else 0.35
        self.phase = 0
        self.grip_count = 0
        self.target_xy = target_xy

    def act(self, obs):
        eef = obs['robot0_eef_pos']
        can = obs['Can_pos']
        grip = -1.0  # open
        target = eef
        HOVER = 1.05
        LIFT = 1.10 if self.arc == 'direct' else 1.25
        if self.phase == 0:      # sideways staging point
            target = can + np.array([0.0, self.side * 0.10, 0.12])
            if np.linalg.norm(eef - target) < 0.03:
                self.phase = 1
        elif self.phase == 1:    # center above can
            target = can + np.array([0.0, 0.0, 0.10])
            if np.linalg.norm(eef[:2] - can[:2]) < 0.012:
                self.phase = 2
        elif self.phase == 2:    # descend
            target = can + np.array([0.0, 0.0, 0.005])
            if eef[2] - can[2] < 0.012:
                self.phase = 3
        elif self.phase == 3:    # close gripper
            target = eef
            grip = 1.0
            self.grip_count += 1
            if self.grip_count > 8:
                self.phase = 4
        elif self.phase == 4:    # lift
            grip = 1.0
            target = np.array([eef[0], eef[1], LIFT])
            if eef[2] > LIFT - 0.02:
                self.phase = 5
        elif self.phase == 5:    # transport
            grip = 1.0
            target = np.array([self.target_xy[0], self.target_xy[1], LIFT])
            if np.linalg.norm(eef[:2] - self.target_xy) < 0.02:
                self.phase = 6
        elif self.phase == 6:    # descend over bin
            grip = 1.0
            target = np.array([self.target_xy[0], self.target_xy[1], 0.98])
            if eef[2] < 1.00:
                self.phase = 7
        else:                    # release
            grip = -1.0
            target = np.array([eef[0], eef[1], HOVER])
        d = np.clip(self.kp * (target - eef), -self.max_step, self.max_step)
        return np.concatenate([d, np.zeros(3), [grip]])


def rollout(side, arc, speed, seed, max_steps=400, record=False):
    env = make_env(seed)
    obs = env.reset()
    ex = Expert(side, arc, speed, target_xy=can_target_xy(env))
    frames, states, actions = [], [], []
    for t in range(max_steps):
        a = ex.act(obs)
        if record:
            frames.append(obs['agentview_image'][::-1].copy())
            states.append(np.concatenate([obs['robot0_eef_pos'], obs['Can_pos'],
                                          [obs['robot0_gripper_qpos'][0]]]))
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
    parser.add_argument('--side', default='left')
    parser.add_argument('--arc', default='direct')
    parser.add_argument('--speed', default='fast')
    args = parser.parse_args()
    ok = 0
    lens = []
    for i in range(args.episodes):
        s, t, *_ = rollout(args.side, args.arc, args.speed, seed=100 + i)
        ok += s
        lens.append(t)
        print(f'ep{i}: success={s} steps={t}', flush=True)
    print(f'EXPERT {args.side}/{args.arc}/{args.speed}: {ok}/{args.episodes} '
          f'mean_steps={np.mean(lens):.0f}', flush=True)
