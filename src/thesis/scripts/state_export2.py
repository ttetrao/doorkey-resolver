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
    """BFS che ritorna l'insieme delle celle sul cammino minimo da start a target."""
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
    """Trova tutte le celle raggiungibili da start, dipendentemente dalla porta aperta o chiusa."""
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
    """Calcola le celle sul percorso ottimo: start -> key -> door -> goal."""
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
    """Transizione di stato deterministica per la Value Iteration."""
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


def _value_iteration(gi, gamma=0.99, theta=1e-6):
    w, h = gi["w"], gi["h"]
    gx, gy = gi["goal_pos"]
    kx, ky = gi["key_pos"]
    dx, dy = gi["door_pos"]
    walls = gi["walls"]

    valid_cells = []
    for x in range(w):
        for y in range(h):
            if (x, y) in walls:
                continue
            valid_cells.append((x, y))

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
    actions = [0, 1, 2, 3, 5]

    while True:
        delta = 0.0
        for s in states:
            x, y, d, has_key, door_open = s
            v_old = V[s]

            if (x, y) == (gx, gy):
                V[s] = 0.0
                continue

            max_q = -float("inf")
            for a in actions:
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


def run_and_save(
    seed: int,
    csv_rows: list,
    size: int = 8,
    history_length: int = 5,
    max_steps: int = 2000,
):
    env = _make_both(size)
    obs, info = env.reset(seed=seed)
    gi = _grid_info(env)

    V_table = _value_iteration(gi, gamma=0.99)

    views = [env.current_view()]
    stages = [_infer_stage(env)]
    states = [_state_desc(env)]
    vals = [round(V_table.get(states[-1], 0.0), 4)]

    rng = np.random.RandomState(seed)
    no_drop = [a for a in range(env.action_space.n) if a != 4]

    terminated = False
    truncated = False

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
        vals.append(round(V_table.get(states[-1], 0.0), 4))

        if stages[-1] == Stage.ERROR:
            break
        if terminated or truncated:
            break

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

    # --- SALVATAGGIO INITIAL ---
    init_file = out_dir / f"initial_step_0000_{stages[0].value}.txt"
    init_file.write_text(views[0])

    row_id = f"{len(csv_rows) + 1:06d}"
    csv_rows.append(
        [
            row_id,
            seed,
            "initial",
            states[0][0],
            states[0][1],
            states[0][2],
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

    # --- SALVATAGGIO WORST ---
    # Cerchiamo il v_value più basso, ma maggiore di 0 (per escludere dead-end e stati iniziali bloccati)
    # Se non ci sono valori > 0, prendiamo il minimo assoluto (fallback)
    positive_vals = [v for v in vals if v > 0.0]
    
    if positive_vals:
        # Trova il primo indice con il valore positivo minimo
        worst_val = min(positive_vals)
        worst_idx = vals.index(worst_val)
    else:
        # Fallback se tutta la traiettoria è a 0.0 (es. l'agente non è mai riuscito a prendere la chiave)
        worst_idx = int(np.argmin(vals))
        worst_val = vals[worst_idx]
        
    worst_stage = stages[worst_idx].value

    worst_file = out_dir / f"worst_step_{worst_idx:04d}_{worst_stage}.txt"
    worst_file.write_text(views[worst_idx])

    row_id = f"{len(csv_rows) + 1:06d}"
    csv_rows.append(
        [
            row_id,
            seed,
            "worst",
            states[worst_idx][0],
            states[worst_idx][1],
            states[worst_idx][2],
            f"{seed}/{worst_file.name}",
            worst_file.name,
            worst_idx,      # step_start
            worst_idx,      # step_end
            worst_stage,    # event
            1,              # n_maps
            worst_val,      # v_value
            0,              # v_llm
        ]
    )
    print(f"  {worst_file.name} — 1 map (V={worst_val})")

    # --- SALVATAGGIO TRANSITIONS & INTERMEDIATE ---
    for ci in change_indices:
        is_goal_step = ci == len(stages) - 1 and terminated

        if is_goal_step:
            end = ci - 1  # L'ultima mappa è quella PRIMA del goal
        else:
            end = ci  # L'ultima mappa è quella dell'evento

        start = max(0, end - (history_length - 1))  # Finestra di history_length mappe

        event = stages[ci - 1].value
        filename = f"transition_step_{start:04d}_{end:04d}_{event}.txt"
        blocks = []
        for idx in range(start, end + 1):
            blocks.append(f"================================================")
            blocks.append(views[idx])
        (out_dir / filename).write_text("\n".join(blocks))

        row_id = f"{len(csv_rows) + 1:06d}"
        csv_rows.append(
            [
                row_id,
                seed,
                "transition",
                states[end][0],
                states[end][1],
                states[end][2],
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

        # --- SALVATAGGIO FILE INTERMEDIATE (sequenza di n passi prima del cambio stage) ---
        if ci >= 4:
            inter_end = ci - 3
            inter_start = max(0, inter_end - (history_length - 1))
            inter_event = stages[inter_end].value
            inter_filename = (
                f"intermediate_step_{inter_start:04d}_{inter_end:04d}_{inter_event}.txt"
            )

            inter_blocks = []
            for idx in range(inter_start, inter_end + 1):
                inter_blocks.append(f"================================================")
                inter_blocks.append(views[idx])
            (out_dir / inter_filename).write_text("\n".join(inter_blocks))

            row_id = f"{len(csv_rows) + 1:06d}"
            csv_rows.append(
                [
                    row_id,
                    seed,
                    "intermediate",
                    states[inter_end][0],
                    states[inter_end][1],
                    states[inter_end][2],
                    f"{seed}/{inter_filename}",
                    inter_filename,
                    inter_start,
                    inter_end,
                    inter_event,
                    inter_end - inter_start + 1,  # n_maps
                    vals[inter_end],
                    0,
                ]
            )
            print(f"  {inter_filename} — {inter_end - inter_start + 1} maps")

    env.close()
    print(
        f"Seed {seed}: {len(views) - 1} steps, "
        f"{len(change_indices)} forward transitions"
    )


def run_off_track(
    seed: int,
    csv_rows: list,
    size: int = 8,
    history_length: int = 5,
    max_steps: int = 2000,
):
    """Genera deterministicamente 1 traiettoria off_track (random walk) per ogni stage."""
    env = _make_both(size)
    obs, info = env.reset(seed=seed)
    gi = _grid_info(env)

    optimal_route = _compute_optimal_route(gi)
    V_table = _value_iteration(gi, gamma=0.99)

    out_dir = Path(__file__).parent.parent / "files" / str(seed) / "off_track"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = env.unwrapped
    kx, ky = gi["key_pos"]
    dx, dy = gi["door_pos"]
    gx, gy = gi["goal_pos"]

    # Trova celle off_track valide per porta chiusa e porta aperta
    def get_off_track_cells(door_open):
        cells = []
        reachable = _reachable_cells(base.grid, tuple(base.agent_pos), door_open)
        for x, y in reachable:
            if (x, y) in optimal_route:
                continue
            if (x, y) == (kx, ky) or (x, y) == (dx, dy) or (x, y) == (gx, gy):
                continue
            cells.append((x, y))
        return cells

    off_track_closed = get_off_track_cells(door_open=False)
    off_track_open = get_off_track_cells(door_open=True)

    # Per ogni stage, definisce: has_key, door_open, e le celle utilizzabili
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

        # Seleziona una posizione off_track a caso da cui partire
        idx = rng.randint(len(cell_pool))
        px, py = cell_pool[idx]

        # Salva lo stato originale dell'ambiente
        orig_pos = tuple(base.agent_pos)
        orig_dir = base.agent_dir
        orig_carrying = base.carrying
        door_obj = base.grid.get(dx, dy)
        orig_door_open = door_obj.is_open if door_obj else None

        # Modifica lo stato dell'ambiente per la generazione off_track
        base.agent_pos = (px, py)
        base.agent_dir = rng.randint(0, 4)

        key_obj = base.grid.get(kx, ky)
        if has_key:
            base.carrying = key_obj
            if key_obj:
                base.grid.set(kx, ky, None)  # Rimuovi la chiave dalla griglia
        else:
            base.carrying = None

        if door_obj:
            door_obj.is_open = door_open

        # Genera una traiettoria sbagliata (random walk) di history_length mappe
        blocks = []
        vals_window = []
        last_state = None

        for _ in range(history_length):
            view = env.current_view()
            state = _state_desc(env)
            val = round(V_table.get(state, 0.0), 4)
            last_state = state

            blocks.append("================================================")
            blocks.append(view)
            vals_window.append(val)

            # Fai un'azione random tra: gira sx (0), gira dx (1), vai avanti (2)
            # Evitiamo pickup(3) e toggle(5) per non cambiare lo stage per sbaglio
            action = rng.choice([0, 1, 2])
            env.step(action)

        # Salviamo il file della traiettoria
        filename = f"off_track_step_0000_{stage.value}.txt"
        (out_dir / filename).write_text("\n".join(blocks))

        row_id = f"{len(csv_rows) + 1:06d}"
        csv_rows.append(
            [
                row_id,
                seed,
                "off_track",
                last_state[0],
                last_state[1],
                last_state[2],
                f"{seed}/off_track/{filename}",
                filename,
                0,
                0,
                stage.value,
                history_length,
                vals_window[-1],
                0,
            ]
        )
        print(f"  off_track/{filename} — {history_length} maps")
        saved += 1

        # Ripristina lo stato originale dell'ambiente per il prossimo stage
        base.agent_pos = orig_pos
        base.agent_dir = orig_dir
        base.carrying = orig_carrying
        if has_key and key_obj:
            base.grid.set(kx, ky, key_obj)
        if door_obj and orig_door_open is not None:
            door_obj.is_open = orig_door_open

    env.close()
    print(f"  off_track/ — {saved} files (1 per stage)")


def main():
    parser = argparse.ArgumentParser(
        description="Esporta stati di DoorKey con Value Iteration."
    )
    parser.add_argument(
        "--size", type=int, default=8, help="Dimensione della mappa (es. 6, 8, 16)"
    )
    parser.add_argument(
        "--history", type=int, default=5, help="Numero di mappe in un blocco di storia"
    )
    args = parser.parse_args()

    size = args.size
    history_length = args.history

    print(f"Avvio generazione: Size={size}x{size}, History Length={history_length}")

    rng = np.random.RandomState(42)
    seeds = [int(rng.randint(0, 1000)) for _ in range(30)]

    csv_rows_metadata: list = []

    header = [
        "id",
        "seed",
        "type",
        "x",
        "y",
        "agent_dir",
        "path",
        "file",
        "step_start",
        "step_end",
        "event",
        "n_maps",
        "v_value",
        "v_llm",
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
