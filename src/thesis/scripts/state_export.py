#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import gymnasium as gym
import numpy as np
from env.view_wrapper import Stage, DoorKeyViewSystem
from env import doorkey_events as doorev

STAGE_ORDER = {
    Stage.FIND_KEY: 0,
    Stage.OPEN_DOOR: 1,
    Stage.REACH_GOAL: 2,
    Stage.ERROR: 3,
}


def _infer_stage(env):
    has_key = doorev.has_key(env)
    door_open = doorev.door_is_open(env)
    goal = doorev.goal_reached(env)

    if not has_key:
        return Stage.FIND_KEY
    if has_key and not door_open:
        return Stage.OPEN_DOOR
    if has_key and door_open and not goal:
        return Stage.REACH_GOAL
    return Stage.ERROR


def _find_obj(grid, obj_type):
    for x in range(grid.width):
        for y in range(grid.height):
            o = grid.get(x, y)
            if o is not None and o.type == obj_type:
                return (x, y, o)
    return None


def _passable(grid, x, y, closed_ok=False):
    if not (0 <= x < grid.width and 0 <= y < grid.height):
        return False
    cell = grid.get(x, y)
    if cell is None:
        return True
    if cell.type == "wall":
        return False
    if cell.type == "door" and not cell.is_open and not closed_ok:
        return False
    return True


def _bfs(grid, start, targets, closed_ok=False):
    from collections import deque

    q = deque()
    q.append((start[0], start[1], None))
    visited = {start}
    target_set = set(targets)
    while q:
        x, y, first = q.popleft()
        if (x, y) in target_set:
            return 2 if first is None else first
        for d, (dx, dy) in enumerate(DIRS):
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited:
                continue
            if not _passable(grid, nx, ny, closed_ok) and (nx, ny) not in target_set:
                continue
            visited.add((nx, ny))
            q.append((nx, ny, d if first is None else first))
    return None


def _neighbors(grid, pos):
    cells = []
    for dx, dy in DIRS:
        nx, ny = pos[0] + dx, pos[1] + dy
        if _passable(grid, nx, ny, closed_ok=True):
            cells.append((nx, ny))
    return cells


def _heuristic_action(env, stage):
    base = env.unwrapped
    ax, ay = base.agent_pos
    agent_dir = base.agent_dir
    grid = base.grid
    s = stage.value

    if s == "find_key":
        found = _find_obj(grid, "key")
        if found:
            if abs(ax - found[0]) + abs(ay - found[1]) == 1:
                return 3  # pickup
            d = _bfs(grid, (ax, ay), [(found[0], found[1])])
            if d is not None:
                return _turn(agent_dir, d)
        found = _find_obj(grid, "door")
        if found:
            d = _bfs(grid, (ax, ay), _neighbors(grid, (found[0], found[1])))
            if d is not None:
                return _turn(agent_dir, d)

    elif s == "open_door":
        found = _find_obj(grid, "door")
        if found:
            tx, ty, door = found
            if abs(ax - tx) + abs(ay - ty) == 1 and not door.is_open:
                return 5  # toggle
            if not door.is_open:
                d = _bfs(grid, (ax, ay), _neighbors(grid, (tx, ty)))
                if d is not None:
                    return _turn(agent_dir, d)
        found = _find_obj(grid, "goal")
        if found:
            d = _bfs(grid, (ax, ay), [(found[0], found[1])], closed_ok=True)
            if d is not None:
                return _turn(agent_dir, d)

    elif s == "reach_goal":
        found = _find_obj(grid, "goal")
        if found:
            d = _bfs(grid, (ax, ay), [(found[0], found[1])], closed_ok=True)
            if d is not None:
                return _turn(agent_dir, d)

    return None  # fallback to random


def _turn(agent_dir, target_dir):
    if agent_dir == target_dir:
        return 2
    diff = (target_dir - agent_dir) % 4
    return 1 if diff == 1 else 0


def _make_both(size):
    base = gym.make(f"MiniGrid-DoorKey-{size}x{size}-v0")
    return DoorKeyViewSystem(base)


from collections import deque

DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # R, D, L, U


def _grid_info(env):
    base = env.unwrapped
    grid = base.grid
    w, h = grid.width, grid.height
    kw = _find_obj(grid, "key")
    dw = _find_obj(grid, "door")
    gw = _find_obj(grid, "goal")
    walls = set()
    for x in range(w):
        for y in range(h):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "wall":
                walls.add((x, y))
    return {
        "grid": grid,
        "w": w,
        "h": h,
        "key_pos": (kw[0], kw[1]),
        "door_pos": (dw[0], dw[1]),
        "goal_pos": (gw[0], gw[1]),
        "walls": walls,
    }


def _state_desc(env):
    base = env.unwrapped
    x, y = base.agent_pos
    d = base.agent_dir
    return (x, y, d, doorev.has_key(env), doorev.door_is_open(env))


def _next_state(state, action, gi):
    x, y, d, has_key, door_open = state
    kx, ky = gi["key_pos"]
    dx, dy = gi["door_pos"]
    w, h, walls = gi["w"], gi["h"], gi["walls"]

    if action == 0:
        return (x, y, (d - 1) % 4, has_key, door_open)
    if action == 1:
        return (x, y, (d + 1) % 4, has_key, door_open)
    if action == 2:
        fx, fy = x + DIRS[d][0], y + DIRS[d][1]
        if not (0 <= fx < w and 0 <= fy < h):
            return None
        if (fx, fy) in walls:
            return None
        if (fx, fy) == (dx, dy) and not door_open:
            return None
        return (fx, fy, d, has_key, door_open)
    if action == 3:
        fx, fy = x + DIRS[d][0], y + DIRS[d][1]
        if (fx, fy) == (kx, ky) and not has_key:
            return (x, y, d, True, door_open)
        return None
    if action == 5:
        fx, fy = x + DIRS[d][0], y + DIRS[d][1]
        if (fx, fy) == (dx, dy) and not door_open:
            return (x, y, d, has_key, True)
        return None
    return None


def _min_steps(state, gi):
    gx, gy = gi["goal_pos"]
    q = deque([(state, 0)])
    visited = {state}
    while q:
        s, dist = q.popleft()
        if s[0] == gx and s[1] == gy:
            return dist
        for a in (0, 1, 2, 3, 5):
            ns = _next_state(s, a, gi)
            if ns is not None and ns not in visited:
                visited.add(ns)
                q.append((ns, dist + 1))
    return None


def _v(state, gi, gamma=0.99):
    d = _min_steps(state, gi)
    if d is None:
        return 0.0
    return round(gamma**d, 4)


def run_and_save(seed: int, csv_rows: list, size: int = 6, max_steps: int = 2000):
    env = _make_both(size)
    obs, info = env.reset(seed=seed)
    gi = _grid_info(env)

    views = [env.current_view()]
    stages = [_infer_stage(env)]
    states = [_state_desc(env)]
    vals = [_v(states[-1], gi)]

    rng = np.random.RandomState(seed)
    no_drop = [a for a in range(env.action_space.n) if a != 4]

    for step in range(1, max_steps):
        if rng.random() < 0.15:
            action = rng.choice(no_drop)
        else:
            action = _heuristic_action(env, stages[-1])
            if action is None:
                action = rng.choice(no_drop)
        obs, reward, terminated, truncated, info = env.step(action)
        views.append(env.current_view())
        stages.append(_infer_stage(env))
        states.append(_state_desc(env))
        vals.append(_v(states[-1], gi))
        if stages[-1] == Stage.ERROR:
            break
        if terminated or truncated:
            break

    change_indices = []
    for i in range(1, len(stages)):
        if stages[i] in STAGE_ORDER and stages[i - 1] in STAGE_ORDER:
            if STAGE_ORDER[stages[i]] > STAGE_ORDER[stages[i - 1]]:
                change_indices.append(i)

    out_dir = Path(__file__).parent.parent / "files" / str(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_file = out_dir / f"initial_step_0000_{stages[0].value}.txt"
    init_file.write_text(views[0])
    csv_rows.append(
        [
            seed,
            "initial",
            f"{seed}/{init_file.name}",
            init_file.name,
            0,
            0,
            stages[0].value,
            1,
            vals[0],
            0,
        ]
    )
    print(f"  {init_file.name} — 1 map")

    for ci in change_indices:
        start = max(0, ci - 5)
        end = ci
        event = stages[ci - 1].value
        filename = f"transition_step_{start:04d}_{end:04d}_{event}.txt"
        blocks = []
        for idx in range(start, end + 1):
            s = stages[idx].value
            blocks.append(f"================================================")
            blocks.append(views[idx])
        (out_dir / filename).write_text("\n".join(blocks))
        csv_rows.append(
            [
                seed,
                "transition",
                f"{seed}/{filename}",
                filename,
                start,
                end,
                event,
                end - start + 1,
                vals[end],
                0,
            ]
        )
        print(f"  {filename} — {end - start + 1} maps")

    env.close()
    print(
        f"Seed {seed}: {len(views) - 1} steps, "
        f"{len(change_indices)} forward transitions"
    )


def run_noise(seed: int, csv_rows: list, size: int = 6, max_steps: int = 2000):
    env = _make_both(size)
    obs, info = env.reset(seed=seed)
    gi = _grid_info(env)

    out_dir = Path(__file__).parent.parent / "files" / str(seed) / "noise"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed + 999)
    no_drop = [a for a in range(env.action_space.n) if a != 4]

    views = [env.current_view()]
    stages = [_infer_stage(env)]
    states = [_state_desc(env)]
    vals = [_v(states[-1], gi)]

    for step in range(1, max_steps):
        action = rng.choice(no_drop)
        obs, reward, terminated, truncated, info = env.step(action)
        views.append(env.current_view())
        stages.append(_infer_stage(env))
        states.append(_state_desc(env))
        vals.append(_v(states[-1], gi))
        if terminated or truncated:
            break

    for start in range(0, len(views) - 5, 6):
        end = start + 5
        stage = stages[end].value
        blocks = []
        for i in range(start, end + 1):
            blocks.append("================================")
            blocks.append(views[i])
        filename = f"noise_{start:04d}_{end:04d}_{stage}.txt"
        (out_dir / filename).write_text("\n".join(blocks))
        csv_rows.append(
            [
                seed,
                "noise",
                f"{seed}/noise/{filename}",
                filename,
                start,
                end,
                stage,
                6,
                vals[end],
                0,
            ]
        )

    env.close()
    print(f"  noise/ — {(len(views) - 5) // 6} chunks of 6 maps")


def main():
    rng = np.random.RandomState(42)
    seeds = [int(rng.randint(0, 1000)) for _ in range(30)]

    csv_rows: list = []
    for seed in seeds:
        run_and_save(seed, csv_rows, size=8)
        run_noise(seed, csv_rows, size=8)

    csv_path = Path(__file__).parent.parent / "files" / "metadata.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "seed",
                "type",
                "path",
                "file",
                "step_start",
                "step_end",
                "event",
                "n_maps",
                "v_value",
                "v_llm",
            ]
        )
        w.writerows(csv_rows)
    print(f"\nWrote {len(csv_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
