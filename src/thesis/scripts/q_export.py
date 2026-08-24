#!/usr/bin/env python3
import sys
import argparse
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

DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # R, D, L, U
ACTIONS = [0, 1, 2, 3, 5]
GAMMA = 0.99

ACTION_NAMES = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    5: "toggle",
    None: "null",
}


# ---------------------------------------------------------------------------
# Helpers di base
# ---------------------------------------------------------------------------
def _find_obj(grid, obj_type):
    for x in range(grid.width):
        for y in range(grid.height):
            o = grid.get(x, y)
            if o is not None and o.type == obj_type:
                return (x, y, o)
    return None


def _infer_stage(env):
    has_key = doorev.has_key(env)
    door_open = doorev.door_is_open(env)
    base = env.unwrapped
    ax, ay = base.agent_pos
    gw = _find_obj(base.grid, "goal")
    goal = gw is not None and (ax, ay) == (gw[0], gw[1])
    if goal:
        return Stage.REACH_GOAL
    if not has_key:
        return Stage.FIND_KEY
    if has_key and not door_open:
        return Stage.OPEN_DOOR
    if has_key and door_open:
        return Stage.REACH_GOAL
    return Stage.ERROR


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
    if cell.type == "key":
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


def _bfs_path_cells(grid, start, target, door_open=False):
    from collections import deque

    q = deque()
    q.append(start)
    parent = {start: None}
    while q:
        cur = q.popleft()
        if cur == target:
            path = set()
            c = cur
            while c is not None:
                path.add(c)
                c = parent[c]
            return path
        for dx, dy in DIRS:
            nx, ny = cur[0] + dx, cur[1] + dy
            if (nx, ny) in parent:
                continue
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                continue
            cell = grid.get(nx, ny)
            if cell is not None:
                if cell.type == "wall":
                    continue
                if cell.type == "door" and not door_open and (nx, ny) != target:
                    continue
                if cell.type == "key" and (nx, ny) != target:
                    continue
            parent[(nx, ny)] = cur
            q.append((nx, ny))
    return set()


def _reachable_cells(grid, start, door_open):
    from collections import deque

    q = deque([start])
    visited = {start}
    while q:
        x, y = q.popleft()
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited:
                continue
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                continue
            cell = grid.get(nx, ny)
            if cell:
                if cell.type == "wall":
                    continue
                if cell.type == "door" and not door_open:
                    continue
                if cell.type == "key":
                    continue
            visited.add((nx, ny))
            q.append((nx, ny))
    return visited


def _compute_optimal_route(gi):
    grid = gi["grid"]
    start = gi["start_pos"]
    kx, ky = gi["key_pos"]
    dx, dy = gi["door_pos"]
    gx, gy = gi["goal_pos"]
    route = set()
    route |= _bfs_path_cells(grid, start, (kx, ky), door_open=False)
    route |= _bfs_path_cells(grid, (kx, ky), (dx, dy), door_open=False)
    route |= _bfs_path_cells(grid, (dx, dy), (gx, gy), door_open=True)
    return route


def _neighbors(grid, pos):
    cells = []
    for dx, dy in DIRS:
        nx, ny = pos[0] + dx, pos[1] + dy
        if _passable(grid, nx, ny, closed_ok=True):
            cells.append((nx, ny))
    return cells


def _turn(agent_dir, target_dir):
    if agent_dir == target_dir:
        return 2
    diff = (target_dir - agent_dir) % 4
    return 1 if diff == 1 else 0


def _make_both(size):
    base = gym.make(f"MiniGrid-DoorKey-{size}x{size}-v0")
    return DoorKeyViewSystem(base)


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
        "start_pos": tuple(base.agent_pos),
    }


def _state_desc(env):
    base = env.unwrapped
    x, y = base.agent_pos
    d = base.agent_dir
    return (x, y, d, doorev.has_key(env), doorev.door_is_open(env))


def _get_next_state(state, action, gi):
    x, y, d, has_key, door_open = state
    kx, ky = gi["key_pos"]
    dx, dy = gi["door_pos"]
    gx, gy = gi["goal_pos"]
    w, h = gi["w"], gi["h"]
    walls = gi["walls"]

    if action == 0:
        return (x, y, (d - 1) % 4, has_key, door_open)
    if action == 1:
        return (x, y, (d + 1) % 4, has_key, door_open)

    if action == 2:
        fx, fy = x + DIRS[d][0], y + DIRS[d][1]
        if not (0 <= fx < w and 0 <= fy < h):
            return state
        if (fx, fy) in walls:
            return state
        if (fx, fy) == (dx, dy) and not door_open:
            return state
        if (fx, fy) == (kx, ky) and not has_key:
            return state
        return (fx, fy, d, has_key, door_open)

    if action == 3:
        fx, fy = x + DIRS[d][0], y + DIRS[d][1]
        if (fx, fy) == (kx, ky) and not has_key:
            return (x, y, d, True, door_open)
        return state

    if action == 5:
        fx, fy = x + DIRS[d][0], y + DIRS[d][1]
        if (fx, fy) == (dx, dy):
            if has_key and not door_open:
                return (x, y, d, has_key, True)
            if door_open:
                return (x, y, d, has_key, False)
        return state

    return state


# ---------------------------------------------------------------------------
# Value Iteration + Q-function
# ---------------------------------------------------------------------------
def _value_iteration(gi, gamma=GAMMA, theta=1e-6):
    w, h = gi["w"], gi["h"]
    gx, gy = gi["goal_pos"]
    kx, ky = gi["key_pos"]
    dx, dy = gi["door_pos"]
    walls = gi["walls"]

    valid_cells = [(x, y) for x in range(w) for y in range(h) if (x, y) not in walls]

    states = []
    for x, y in valid_cells:
        for d in range(4):
            for has_key in [False, True]:
                for door_open in [False, True]:
                    if not has_key and (x, y) == (kx, ky):
                        continue
                    if not door_open and (x, y) == (dx, dy):
                        continue
                    states.append((x, y, d, has_key, door_open))

    V = {s: 0.0 for s in states}

    while True:
        delta = 0.0
        for s in states:
            x, y, d, has_key, door_open = s
            v_old = V[s]
            if (x, y) == (gx, gy):
                V[s] = 0.0
                continue
            max_q = -float("inf")
            for a in ACTIONS:
                ns = _get_next_state(s, a, gi)
                r = 1.0 if (ns[0], ns[1]) == (gx, gy) and (x, y) != (gx, gy) else 0.0
                q = r + gamma * V.get(ns, 0.0)
                if q > max_q:
                    max_q = q
            V[s] = max_q if max_q != -float("inf") else 0.0
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    return V


def _q_value(state, action, gi, V, gamma=GAMMA):
    """Q(s,a) = r + gamma * V(s') usando la V-table già calcolata."""
    ns = _get_next_state(state, action, gi)
    x, y = state[0], state[1]
    gx, gy = gi["goal_pos"]
    r = 1.0 if (ns[0], ns[1]) == (gx, gy) and (x, y) != (gx, gy) else 0.0
    return round(r + gamma * V.get(ns, 0.0), 6)


def _v_value(state, V):
    return round(V.get(state, 0.0), 6)


def _greedy_action(state, gi, V):
    """Azione greedy rispetto a V (fallback se non c'è azione registrata)."""
    best_a, best_q = None, -float("inf")
    for a in ACTIONS:
        ns = _get_next_state(state, a, gi)
        x, y = state[0], state[1]
        gx, gy = gi["goal_pos"]
        r = 1.0 if (ns[0], ns[1]) == (gx, gy) and (x, y) != (gx, gy) else 0.0
        q = r + GAMMA * V.get(ns, 0.0)
        if q > best_q:
            best_q = q
            best_a = a
    return best_a if best_a is not None else 2


# ---------------------------------------------------------------------------
# Heuristic policy
# ---------------------------------------------------------------------------
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
                return 3
            d = _bfs(grid, (ax, ay), _neighbors(grid, (found[0], found[1])))
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
                return 5
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

    return None


# ---------------------------------------------------------------------------
# Helper per scrivere le righe CSV per Stato (s) e Azioni (s,a)
# ---------------------------------------------------------------------------
def _append_q_rows(
    csv_rows,
    seed,
    type_str,
    path_str,
    file_str,
    step_start,
    step_end,
    event,
    n_maps,
    state,
    V,
    gi,
):
    """Emette 1 riga (s) con entry_type=0 e 5 righe (s,a) con entry_type=1 per ogni azione."""
    v_val = _v_value(state, V)

    # 1. entry_type=0: solo stato (s)
    row_id_0 = f"{len(csv_rows) + 1:06d}"
    csv_rows.append(
        [
            row_id_0,
            seed,
            type_str,
            path_str,
            file_str,
            0,  # entry_type
            "null",  # action
            step_start,
            step_end,
            event,
            n_maps,
            v_val,  # v_value
            0.0,  # q_value
            0,  # q_llm
        ]
    )

    # 2. entry_type=1: stato + azione (s,a) per ogni azione possibile
    for a in ACTIONS:
        q_val = _q_value(state, a, gi, V)
        act_name = ACTION_NAMES.get(a, "null")

        row_id_1 = f"{len(csv_rows) + 1:06d}"
        csv_rows.append(
            [
                row_id_1,
                seed,
                type_str,
                path_str,
                file_str,
                1,  # entry_type
                act_name,  # action
                step_start,
                step_end,
                event,
                n_maps,
                0.0,  # v_value (informativo, 0 per pulizia)
                q_val,  # q_value
                0,  # q_llm
            ]
        )


# ---------------------------------------------------------------------------
# Run & save (worst, transition, intermediate)
# ---------------------------------------------------------------------------
def run_and_save(seed, csv_rows, size=8, history_length=5, max_steps=2000):
    env = _make_both(size)
    obs, info = env.reset(seed=seed)
    gi = _grid_info(env)
    V = _value_iteration(gi, gamma=GAMMA)

    views = [env.current_view()]
    stages = [_infer_stage(env)]
    states = [_state_desc(env)]

    rng = np.random.RandomState(seed)
    no_drop = [a for a in range(env.action_space.n) if a != 4]

    terminated = False
    truncated = False

    for step in range(1, max_steps):
        if rng.random() < 0.15:
            action = int(rng.choice(no_drop))
        else:
            action = _heuristic_action(env, stages[-1])
            if action is None:
                action = int(rng.choice(no_drop))

        obs, reward, terminated, truncated, info = env.step(action)

        views.append(env.current_view())
        stages.append(_infer_stage(env))
        states.append(_state_desc(env))

        if stages[-1] == Stage.ERROR:
            break
        if terminated or truncated:
            break

    # cambio stage
    change_indices = []
    for i in range(1, len(stages)):
        if stages[i] in STAGE_ORDER and stages[i - 1] in STAGE_ORDER:
            if STAGE_ORDER[stages[i]] > STAGE_ORDER[stages[i - 1]]:
                change_indices.append(i)
    if terminated and len(stages) > 1:
        if (len(stages) - 1) not in change_indices:
            change_indices.append(len(stages) - 1)

    out_dir = Path(__file__).parent.parent / "files" / str(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- WORST ----------
    vals = [_v_value(s, V) for s in states]
    positive_vals = [v for v in vals if v > 0.0]
    if positive_vals:
        worst_val = min(positive_vals)
        worst_idx = vals.index(worst_val)
    else:
        worst_idx = int(np.argmin(vals))
        worst_val = vals[worst_idx]

    worst_stage = stages[worst_idx].value

    worst_file = out_dir / f"worst_step_{worst_idx:04d}_{worst_stage}.txt"
    worst_file.write_text(views[worst_idx])

    _append_q_rows(
        csv_rows,
        seed=seed,
        type_str="worst",
        path_str=f"{seed}/{worst_file.name}",
        file_str=worst_file.name,
        step_start=worst_idx,
        step_end=worst_idx,
        event=worst_stage,
        n_maps=1,
        state=states[worst_idx],
        V=V,
        gi=gi,
    )
    print(f"  {worst_file.name} — 1 map (V={worst_val})")

    # ---------- TRANSITION & INTERMEDIATE ----------
    for ci in change_indices:
        # Lo stato a 1 azione di distanza dal cambio stage è sempre ci-1.
        # Per le transizioni normali è l'azione che causa il cambio.
        # Per il goal (reach_goal) è l'azione che porta sul goal (come da script originale).
        eval_idx = ci - 1

        if eval_idx < 0:
            continue

        start = max(0, eval_idx - (history_length - 1))
        event = stages[eval_idx].value

        filename = f"transition_step_{start:04d}_{eval_idx:04d}_{event}.txt"
        blocks = []
        for idx in range(start, eval_idx + 1):
            blocks.append("================================================")
            blocks.append(views[idx])
        (out_dir / filename).write_text("\n".join(blocks))

        _append_q_rows(
            csv_rows,
            seed=seed,
            type_str="transition",
            path_str=f"{seed}/{filename}",
            file_str=filename,
            step_start=start,
            step_end=eval_idx,
            event=event,
            n_maps=eval_idx - start + 1,
            state=states[eval_idx],
            V=V,
            gi=gi,
        )
        print(f"  {filename} — {eval_idx - start + 1} maps")

        # ---------- INTERMEDIATE ----------
        # L'intermediate è 3 passi prima del transition, quindi eval_idx - 3
        inter_eval_idx = eval_idx - 3
        if inter_eval_idx >= 0:
            inter_start = max(0, inter_eval_idx - (history_length - 1))
            inter_event = stages[inter_eval_idx].value
            inter_filename = f"intermediate_step_{inter_start:04d}_{inter_eval_idx:04d}_{inter_event}.txt"
            inter_blocks = []
            for idx in range(inter_start, inter_eval_idx + 1):
                inter_blocks.append("================================================")
                inter_blocks.append(views[idx])
            (out_dir / inter_filename).write_text("\n".join(inter_blocks))

            _append_q_rows(
                csv_rows,
                seed=seed,
                type_str="intermediate",
                path_str=f"{seed}/{inter_filename}",
                file_str=inter_filename,
                step_start=inter_start,
                step_end=inter_eval_idx,
                event=inter_event,
                n_maps=inter_eval_idx - inter_start + 1,
                state=states[inter_eval_idx],
                V=V,
                gi=gi,
            )
            print(f"  {inter_filename} — {inter_eval_idx - inter_start + 1} maps")

    env.close()


# ---------------------------------------------------------------------------
# Off-track
# ---------------------------------------------------------------------------
def run_off_track(seed, csv_rows, size=8, history_length=5, max_steps=2000):
    env = _make_both(size)
    obs, info = env.reset(seed=seed)
    gi = _grid_info(env)
    optimal_route = _compute_optimal_route(gi)
    V = _value_iteration(gi, gamma=GAMMA)

    out_dir = Path(__file__).parent.parent / "files" / str(seed) / "off_track"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = env.unwrapped
    kx, ky = gi["key_pos"]
    dx, dy = gi["door_pos"]
    gx, gy = gi["goal_pos"]

    def get_off_track_cells(door_open):
        cells = []
        reachable = _reachable_cells(base.grid, tuple(base.agent_pos), door_open)
        for x, y in reachable:
            if (x, y) in optimal_route:
                continue
            if (x, y) in [(kx, ky), (dx, dy), (gx, gy)]:
                continue
            cells.append((x, y))
        return cells

    off_track_closed = get_off_track_cells(door_open=False)
    off_track_open = get_off_track_cells(door_open=True)

    stages_to_generate = [
        (Stage.FIND_KEY, False, False, off_track_closed),
        (Stage.OPEN_DOOR, True, False, off_track_closed),
        (Stage.REACH_GOAL, True, True, off_track_open),
    ]

    rng = np.random.RandomState(seed + 999)
    saved = 0

    for stage, has_key, door_open, cell_pool in stages_to_generate:
        if not cell_pool:
            continue

        idx = rng.randint(len(cell_pool))
        px, py = cell_pool[idx]

        orig_pos = tuple(base.agent_pos)
        orig_dir = base.agent_dir
        orig_carrying = base.carrying
        door_obj = base.grid.get(dx, dy)
        orig_door_open = door_obj.is_open if door_obj else None

        base.agent_pos = (px, py)
        base.agent_dir = int(rng.randint(0, 4))

        key_obj = base.grid.get(kx, ky)
        if has_key:
            base.carrying = key_obj
            if key_obj:
                base.grid.set(kx, ky, None)
        else:
            base.carrying = None
        if door_obj:
            door_obj.is_open = door_open

        blocks = []
        first_state = None

        for step_i in range(history_length):
            view = env.current_view()
            state = _state_desc(env)
            if first_state is None:
                first_state = state

            blocks.append("================================================")
            blocks.append(view)

            action = int(rng.choice([0, 1, 2]))
            env.step(action)

        filename = f"off_track_step_0000_{stage.value}.txt"
        (out_dir / filename).write_text("\n".join(blocks))

        _append_q_rows(
            csv_rows,
            seed=seed,
            type_str="off_track",
            path_str=f"{seed}/off_track/{filename}",
            file_str=filename,
            step_start=0,
            step_end=0,
            event=stage.value,
            n_maps=history_length,
            state=first_state,
            V=V,
            gi=gi,
        )
        print(f"  off_track/{filename} — {history_length} maps")
        saved += 1

        # restore env
        base.agent_pos = orig_pos
        base.agent_dir = orig_dir
        base.carrying = orig_carrying
        if has_key and key_obj:
            base.grid.set(kx, ky, key_obj)
        if door_obj and orig_door_open is not None:
            door_obj.is_open = orig_door_open

    env.close()
    print(f"  off_track/ — {saved} files (1 per stage)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Esporta stati e Q-function di DoorKey con Value Iteration."
    )
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--history", type=int, default=5)
    args = parser.parse_args()

    size = args.size
    history_length = args.history
    print(f"Avvio generazione: Size={size}x{size}, History Length={history_length}")

    rng = np.random.RandomState(42)
    seeds = [int(rng.randint(0, 1000)) for _ in range(30)]

    csv_rows_metadata = []
    header = [
        "id",
        "seed",
        "type",
        "path",
        "file",
        "entry_type",
        "action",
        "step_start",
        "step_end",
        "event",
        "n_maps",
        "v_value",
        "q_value",
        "q_llm",
    ]

    for seed in seeds:
        run_and_save(seed, csv_rows_metadata, size=size, history_length=history_length)
        run_off_track(seed, csv_rows_metadata, size=size, history_length=history_length)

    csv_path_meta = Path(__file__).parent.parent / "files" / "metadata.csv"
    csv_path_meta.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path_meta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(csv_rows_metadata)
    print(f"\nWrote {len(csv_rows_metadata)} rows to {csv_path_meta}")


if __name__ == "__main__":
    main()
