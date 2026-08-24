import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from minigrid.manual_control import ManualControl

from env.view_wrapper import DoorKeyRewardSystem


class ViewPrintWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        print(self.env.current_view())
        print()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        print(self.env.current_view())
        print()
        return obs, reward, terminated, truncated, info


class CustomManualControl(ManualControl):
    def key_handler(self, event):
        key = event.key
        actions = self.env.unwrapped.actions

        if key == "escape":
            self.env.close()
        elif key == "backspace":
            self.reset(self.seed)
        elif key == "left":
            self.step(actions.left)
        elif key == "right":
            self.step(actions.right)
        elif key == "up":
            self.step(actions.forward)
        elif key == "space":
            self.step(actions.toggle)
        elif key == "p":
            self.step(actions.pickup)
        elif key == "d":
            self.step(actions.drop)
        else:
            super().key_handler(event)


def main():
    parser = argparse.ArgumentParser(description="DoorKey interactive view explorer")
    parser.add_argument(
        "--size", type=int, default=5, choices=[5, 6, 8, 16],
        help="Env grid size (default: 5)"
    )
    args = parser.parse_args()

    env_id = f"MiniGrid-DoorKey-{args.size}x{args.size}-v0"
    env = gym.make(env_id, render_mode="human")
    env = FullyObsWrapper(env)
    env = DoorKeyRewardSystem(env)
    env = ViewPrintWrapper(env)

    print(f"\n=== DoorKey {args.size}x{args.size} ===")
    print("CONTROLS: UP=fwd  LEFT/RIGHT=turn  SPACE=toggle  P=pickup  D=drop  ESC=quit\n")

    mc = CustomManualControl(env, seed=42)
    mc.start()


if __name__ == "__main__":
    main()
