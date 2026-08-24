# Door Key

![Door Key](https://minigrid.farama.org/_images/DoorKeyEnv.gif)

|                   |                                                                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Action Space      | `Discrete(7)`                                                                                                                                                 |
| Observation Space | `Dict('direction': Discrete(4), 'image': Box(0, 255, (7, 7, 3), uint8), 'mission': MissionSpace(<function DoorKeyEnv._gen_mission at 0x7f69ef6228e0>, None))` |
| Creation          | `gymnasium.make("MiniGrid-DoorKey-16x16-v0")`                                                                                                                 |

## Description

This environment has a key that the agent must pick up in order to unlock a door and then get to the green goal square. This environment is difficult, because of the sparse reward, to solve using classical RL algorithms. It is useful to experiment with curiosity or curriculum learning.

## Mission Space

“use the key to open the door and then get to the goal”

## Action Space

| Num | Name    | Action                    |
| --- | ------- | ------------------------- |
| 0   | left    | Turn left                 |
| 1   | right   | Turn right                |
| 2   | forward | Move forward              |
| 3   | pickup  | Pick up an object         |
| 4   | drop    | Unused                    |
| 5   | toggle  | Toggle/activate an object |
| 6   | done    | Unused                    |

to pickup the key you need to use the action "pickup" in an adjacent position while facing the key.

## Observation Encoding

- Each tile is encoded as a 3 dimensional tuple: `(OBJECT_IDX, COLOR_IDX, STATE)`
- `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in [minigrid/core/constants.py](https://minigrid.farama.org/_modules/minigrid/core/constants/)
- `STATE` refers to the door state with 0=open, 1=closed and 2=locked

## Rewards

A reward of ‘1 - 0.9 \* (step_count / max_steps)’ is given for success, and ‘0’ for failure only when you reach the goal.

## Termination

The episode ends if any one of the following conditions is met:

1. The agent reaches the goal.
2. Timeout (see `max_steps`).
