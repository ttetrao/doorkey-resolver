import gymnasium as gym

from .rewardsystem import DoorKeyRewardSystem, RewardConfig
from .view_wrapper import DoorKeyViewSystem


def make_env(render_mode=None, reward_config=None, size=6, debug=False):
    env_id = f"MiniGrid-DoorKey-{size}x{size}-v0"
    env = gym.make(env_id, render_mode=render_mode)

    if debug:
        if reward_config is None:
            reward_config = RewardConfig()
        env = DoorKeyRewardSystem(env, reward_config)
    else:
        env = DoorKeyViewSystem(env)
    return env
