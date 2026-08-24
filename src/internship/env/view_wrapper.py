from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from os import eventfd
from typing import Any, cast
from collections import deque
from enum import IntEnum
import gymnasium as gym
from minigrid.core.world_object import Door
from minigrid.minigrid_env import MiniGridEnv

from . import doorkey_events as doorev

LEGEND = """
+----+----+----+----+
|    |A(U)| ▇  | ▇  |
+----+----+----+----+
|    | K  | ▇  | G  |
+----+----+----+----+
|    |    |D(L)|    |
+----+----+----+----+
=====================
+----+----+----+----+
|    |    | ▇  | ▇  |
+----+----+----+----+
|    |L(D)| ▇  | G  |
+----+----+----+----+
|    |    |D(L)|    |
+----+----+----+----+
====================
+----+----+----+----+
|    |    | ▇  | ▇  |
+----+----+----+----+
|    |    | ▇  | G  |
+----+----+----+----+
|    |L(L)|D(O)|    |
+----+----+----+----+
====================
...

Legend:

A(U) = Agent (up)
A(D) = Agent (down)
A(R) = Agent (right)
A(L) = Agent (left)

K = Key

L(U) = Agent (loaded) (up)
L(D) = Agent (loaded) (down)
L(R) = Agent (loaded) (right)
L(L) = Agent (loaded) (left)


▇ = wall (not traversable)

D = door (traversable, can be opened only by L - agent with key)

D(L) = door (locked)
D(O) = door (open)


Stage:
FIND_KEY = "find_key"
OPEN_DOOR = "open_door"
REACH_GOAL = "reach_goal"

"""


# ==========================================
# CANALE 0: Tipi di Oggetti
# ==========================================
class ObjectType(IntEnum):
    EMPTY = 1
    WALL = 2
    DOOR = 4
    KEY = 5
    GOAL = 8
    AGENT = 10


# ==========================================
# CANALE 2: Stati (Principalmente per le porte)
# ==========================================
class DoorState(IntEnum):
    OPEN = 0
    CLOSED = 1
    LOCKED = 2


TRANSLATE_OBJ = {
    ObjectType.EMPTY: "   ",
    ObjectType.DOOR: " D ",
    ObjectType.KEY: " K ",
    ObjectType.GOAL: " G ",
    ObjectType.WALL: " ▇ ",
    ObjectType.AGENT: " A ",
}


# ─────────────────────────────────────────────
# Enum che rappresenta le fasi sequenziali del task DoorKey.
# ─────────────────────────────────────────────
class Stage(Enum):
    FIND_KEY = "find_key"
    OPEN_DOOR = "open_door"
    REACH_GOAL = "reach_goal"
    ERROR = "error"


# ─────────────────────────────────────────────
# Snapshot degli eventi booleani rilevanti in un dato timestep.
# ─────────────────────────────────────────────
@dataclass
class EventSnapshot:
    has_key: bool
    door_is_open: bool
    goal_reached: bool

    def __iter__(self):
        return iter((self.has_key, self.door_is_open, self.goal_reached))


class DoorKeyViewSystem(gym.Wrapper):
    def __init__(self, env: gym.Env):
        self.current_stage: Stage | None = None
        self.curr_events: EventSnapshot | None = None

        self.key_pos: tuple[int, int] | None = None
        self.door_pos: tuple[int, int] | None = None
        self.goal_pos: tuple[int, int] | None = None

        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.key_pos = self._find_stage_goal_position("key")
        self.door_pos = self._find_stage_goal_position("door")
        self.goal_pos = self._find_stage_goal_position("goal")

        base_env = self._get_base_env()
        self.curr_events = self._extract_events()
        self.curr_stage = self._infer_stage(self.curr_events)

        return obs, info

    def current_view(self):
        base_env = self._get_base_env()
        grid = base_env.grid
        width, height = grid.width, grid.height

        ax, ay = base_env.agent_pos
        agent_dir = base_env.agent_dir
        carrying = base_env.carrying
        has_key = carrying is not None and carrying.type == "key"

        dir_sym = ["R", "D", "L", "U"]
        prefix = "L" if has_key else "A"
        agent_sym = f"{prefix}({dir_sym[agent_dir]})"

        cell_sym = {
            "wall": " ▇  ",
            "door": " D  ",
            "key": " K  ",
            "goal": " G  ",
        }

        x_start, x_end = 1, width - 1
        y_start, y_end = 1, height - 1
        inner_w = x_end - x_start
        inner_h = y_end - y_start

        lines = []
        border = "+" + "----+" * inner_w

        for y in range(y_start, y_end):
            lines.append(border)
            row = []
            for x in range(x_start, x_end):
                if (x, y) == (ax, ay):
                    row.append(agent_sym)
                else:
                    cell = grid.get(x, y)
                    if cell is None:
                        sym = "    "
                    elif cell.type == "door":
                        door = cast(Door, cell)
                        sym = "D(O)" if door.is_open else "D(C)"
                    else:
                        sym = cell_sym.get(cell.type, "    ")
                    row.append(sym)
            lines.append("|" + "|".join(row) + "|")
        lines.append(border)

        return "\n".join(lines)

    def step(self, action):
        if self.curr_events is None or self.curr_stage is None:
            raise RuntimeError(
                "Wrapper state not initialized. Call reset() before step()."
            )

        obs, env_reward, terminated, truncated, info = self.env.step(action)

        return obs, env_reward, terminated, truncated, info

    def _get_base_env(self) -> MiniGridEnv:
        return cast(MiniGridEnv, self.env.unwrapped)

    def _extract_events(self) -> EventSnapshot:
        return EventSnapshot(
            has_key=doorev.has_key(self),
            door_is_open=doorev.door_is_open(self),
            goal_reached=doorev.goal_reached(self),
        )

    def _infer_stage(self, events: EventSnapshot) -> Stage:
        if not events.has_key:
            return Stage.FIND_KEY
        elif events.has_key and not events.door_is_open:
            return Stage.OPEN_DOOR
        elif events.has_key and events.door_is_open and not events.goal_reached:
            return Stage.REACH_GOAL
        else:
            return Stage.ERROR

    def _find_stage_goal_position(self, goal) -> tuple[int, int]:
        base_env = self._get_base_env()
        grid = base_env.grid
        for x in range(grid.width):
            for y in range(grid.height):
                obj = grid.get(x, y)
                if obj is not None and obj.type == goal:
                    return (x, y)
        raise RuntimeError(goal + " not found in the grid")
