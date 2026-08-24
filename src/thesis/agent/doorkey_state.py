#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from env.view_wrapper import Stage

STAGE_IDX = {Stage.FIND_KEY: 0, Stage.OPEN_DOOR: 1, Stage.REACH_GOAL: 2, Stage.ERROR: 2}
EVENT_STAGE = {"find_key": 0, "open_door": 1, "reach_goal": 2}


def encode(env):
    stage = env.get_wrapper_attr("curr_stage")
    stage_idx = STAGE_IDX.get(stage, 2)
    if stage == Stage.FIND_KEY:
        t = env.get_wrapper_attr("key_pos")
    elif stage == Stage.OPEN_DOOR:
        t = env.get_wrapper_attr("door_pos")
    else:
        t = env.get_wrapper_attr("goal_pos")
    base = env.unwrapped
    ax, ay = base.agent_pos
    return (int(t[0] - ax), int(t[1] - ay), int(base.agent_dir), int(stage_idx))


def to_vec(st):
    dx, dy, d, stage = st
    return np.array([dx / 7.0, dy / 7.0, d / 3.0, stage / 2.0], dtype=np.float32)


def build_known(csv_path, meta_path, env):
    meta = {}
    for r in csv.DictReader(open(meta_path)):
        meta[(int(r["seed"]), r["file"])] = (int(r["x"]), int(r["y"]), int(r["agent_dir"]))
    pos = {}
    agg = {}
    for r in csv.DictReader(open(csv_path)):
        seed, f = int(r["seed"]), r["file"]
        if (seed, f) not in meta:
            continue
        if seed not in pos:
            env.reset(seed=seed)
            pos[seed] = (env.get_wrapper_attr("key_pos"),
                         env.get_wrapper_attr("door_pos"),
                         env.get_wrapper_attr("goal_pos"))
        key_pos, door_pos, goal_pos = pos[seed]
        t = (key_pos, door_pos, goal_pos)[EVENT_STAGE[r["event"]]]
        x, y, d = meta[(seed, f)]
        st = (int(t[0] - x), int(t[1] - y), int(d), EVENT_STAGE[r["event"]])
        v = float(r["v_llm"])
        if st in agg:
            agg[st] = (agg[st][0] + v, agg[st][1] + 1)
        else:
            agg[st] = (v, 1)
    return {st: v / n for st, (v, n) in agg.items()}


def build_potential(known, k=3, cutoff=6):
    """Potenziale v_llm: identico a known sugli stati noti, altrimenti
    media pesata dei k stati noti piu' vicini nello stesso stage.
    Con known={} restituisce un potenziale identicamente zero."""
    per_stage = defaultdict(list)
    for (dx, dy, d, sg), v in known.items():
        per_stage[sg].append((dx, dy, d, v))

    def cyclic_dir(a, b):
        d = abs(a - b) % 4
        return min(d, 4 - d)

    def phi(st):
        if st in known:
            return known[st]
        dx, dy, d, sg = st
        ds = [(abs(cx - dx) + abs(cy - dy) + cyclic_dir(d, cd), v)
              for cx, cy, cd, v in per_stage[sg]]
        ds = [(dd, v) for dd, v in ds if dd <= cutoff]
        if not ds:
            return 0.0
        ds.sort(key=lambda t: t[0])
        wsum = 0.0
        vsum = 0.0
        for dd, v in ds[:k]:
            w = 1.0 / (dd + 1.0)
            wsum += w
            vsum += w * v
        return vsum / wsum

    return phi


def plot_compare(hist_v, hist_a, eval_v, eval_a, out_name="ddqn_compare.png", window=100):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, key, title in ((axes[0], "successes", "Success Rate (media mobile)"),
                           (axes[1], "rewards", "Reward (media mobile, negativi tagliati)"),
                           (axes[2], "losses", "Loss")):
        for hist, label in ((hist_v, "Vanilla"), (hist_a, "Augmented")):
            ax.plot(hist[key], alpha=0.3)
            if len(hist[key]) >= window:
                ma = np.convolve(hist[key], np.ones(window) / window, mode='valid')
                ax.plot(range(window - 1, len(hist[key])), ma, label=label, linewidth=2)
            else:
                ax.plot([], label=label)
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1.05)
    axes[0].text(0.02, 0.92, f"eval SR: vanilla={eval_v:.3f}  augmented={eval_a:.3f}",
                 transform=axes[0].transAxes, fontsize=10,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.suptitle("Confronto Vanilla vs Augmented (v_llm shaping)")
    plt.tight_layout()
    out = Path(__file__).parent.parent / out_name
    fig.savefig(out, dpi=150)
    print(f"Plot salvato: {out}")
    plt.close(fig)