#!/usr/bin/env python3
import argparse
import csv
import random
import sys
from pathlib import Path
from collections import defaultdict, deque
from typing import cast

import numpy as np
import gymnasium as gym
import minigrid
from gymnasium.spaces import Discrete

sys.path.insert(0, str(Path(__file__).parent.parent))
from env.view_wrapper import DoorKeyViewSystem
from doorkey_state import encode, build_known, build_potential, plot_compare


class QLearningAgent:
    def __init__(self, n_actions, alpha=0.3, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.9985, lam=1.0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lam = lam
        self.q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))

    def act(self, state, greedy=False):
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q[state]))

    def update(self, s, a, r, s_next, done):
        best_next = 0.0 if done else np.max(self.q[s_next])
        td = r + self.gamma * best_next - self.q[s][a]
        self.q[s][a] += self.alpha * td
        return abs(td)

    def anchor(self, s, v):
        """Vincolo: forza max_a Q(s,a) verso V_known(s), solo sull'azione
        correntemente massima (il gradiente del max agisce solo su a*)."""
        a = int(np.argmax(self.q[s]))
        corr = v - self.q[s][a]
        self.q[s][a] += self.alpha * self.lam * corr
        return abs(corr)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(env, agent, episodes, potential, known, csv_seeds, max_steps=0, log_every=100):
    rewards, losses, successes, ep_lengths = [], [], [], []
    sr_buffer = deque(maxlen=100)
    log_hits = 0
    log_phi = 0.0
    log_anchor = 0.0
    log_ahits = 0

    for ep in range(episodes):
        p = 1.0 - 0.8 * ep / max(episodes - 1, 1)
        seed = random.choice(csv_seeds) if csv_seeds and random.random() < p else random.randint(0, 1000)
        env.reset(seed=seed)
        st = encode(env)
        ep_rew = 0.0
        ep_td = 0.0
        steps = 0
        while max_steps == 0 or steps < max_steps:
            a = agent.act(st)
            ns, r, term, trunc, _ = env.step(a)
            nst = encode(env)
            done = term or trunc
            phi_s = potential(st)
            phi_n = potential(nst)
            if phi_s > 0 or phi_n > 0:
                log_hits += 1
                log_phi += phi_s + phi_n
            r += agent.gamma * phi_n - phi_s
            ep_td += agent.update(st, a, r, nst, done)
            if st in known:
                log_anchor += agent.anchor(st, known[st])
                log_ahits += 1
            ep_rew += r
            steps += 1
            st = nst
            if done:
                break
        agent.decay_epsilon()
        is_success = 1.0 if (term and not trunc and ep_rew > 0) else 0.0
        sr_buffer.append(is_success)
        rewards.append(ep_rew)
        losses.append(ep_td / max(steps, 1))
        successes.append(is_success)
        ep_lengths.append(steps)

        if ep % log_every == 0:
            avg_r = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            phi_avg = log_phi / (2 * log_hits) if log_hits else 0.0
            anchor_avg = log_anchor / log_ahits if log_ahits else 0.0
            print(f"Ep {ep:5d}: reward={ep_rew:.2f} avg100={avg_r:.2f} sr={np.mean(sr_buffer):.3f} eps={agent.epsilon:.3f} states={len(agent.q)} v_llm={log_hits} hits φavg={phi_avg:.2f} anchor={log_ahits} hits corr={anchor_avg:.3f}")
            log_hits = 0
            log_phi = 0.0
            log_anchor = 0.0
            log_ahits = 0

    return {"rewards": rewards, "losses": losses, "successes": successes, "ep_lengths": ep_lengths}


def evaluate(env, agent, episodes, seeds):
    sr = []
    for i in range(episodes):
        env.reset(seed=seeds[i])
        st = encode(env)
        ep_rew = 0.0
        while True:
            a = agent.act(st, greedy=True)
            st, r, term, trunc, _ = env.step(a)
            st = encode(env)
            ep_rew += r
            if term or trunc:
                break
        sr.append(1.0 if term and not trunc and ep_rew > 0 else 0.0)
    return np.mean(sr)


def main():
    parser = argparse.ArgumentParser(description="Q-Learning tabulare con vincolo Bellman residual (anchor v_llm)")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_decay", type=float, default=0.9985)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0,
                        help="peso del vincolo (max_a Q(s) -> V_known)")
    parser.add_argument("--shaping", action="store_true",
                        help="aggiunge anche il potenziale interpolato v_llm come reward shaping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", default=str(Path(__file__).parent.parent / "result/8x8 final/output_gemma-4-26b-a4b-it.csv"))
    parser.add_argument("--max_steps", type=int, default=0, help="0 = limite env (640)")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--compare", action="store_true", help="train vanilla e augmented, test e confronta")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    env_id = "MiniGrid-DoorKey-8x8-v0"
    base = gym.make(env_id)
    n_actions = int(cast(Discrete, base.action_space).n)
    base.close()

    env = DoorKeyViewSystem(gym.make(env_id))
    env.reset(seed=args.seed)

    known = build_known(args.csv, str(Path(__file__).parent.parent / "files/metadata.csv"), env)
    with open(args.csv) as f:
        csv_seeds = sorted({int(r["seed"]) for r in csv.DictReader(f)})
    print(f"Known states: {len(known)}, csv seeds: {len(csv_seeds)}")

    if args.selfcheck:
        for s0 in csv_seeds[:3]:
            env.reset(seed=s0)
            assert encode(env) in known, f"initial state of seed {s0} not in known"
        print("Selfcheck OK")
        return

    def make_agent(lam):
        return QLearningAgent(
            n_actions=n_actions,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_decay=args.eps_decay,
            lam=lam,
        )

    if args.compare:
        eval_seeds = random.sample(range(100000), 200)

        random.seed(args.seed); np.random.seed(args.seed)
        agent_v = make_agent(0.0)
        hist_v = train(env, agent_v, args.episodes, build_potential({}), {}, [], args.max_steps, args.log_every)
        eval_v = evaluate(env, agent_v, len(eval_seeds), eval_seeds)
        print(f"Vanilla eval SR: {eval_v:.3f}")

        random.seed(args.seed); np.random.seed(args.seed)
        agent_a = make_agent(args.lam)
        pot_a = build_potential(known) if args.shaping else build_potential({})
        hist_a = train(env, agent_a, args.episodes, pot_a, known, csv_seeds, args.max_steps, args.log_every)
        eval_a = evaluate(env, agent_a, len(eval_seeds), eval_seeds)
        print(f"Augmented eval SR: {eval_a:.3f}")

        plot_compare(hist_v, hist_a, eval_v, eval_a, "qtable_anchor_compare.png")
        env.close()
        return

    agent = make_agent(args.lam)
    pot = build_potential(known) if args.shaping else build_potential({})
    print(f"Training Q-Learning-anchor: {args.episodes} episodes, alpha={args.alpha}, lam={args.lam}, shaping={args.shaping}")
    train(env, agent, args.episodes, pot, known, csv_seeds, args.max_steps, args.log_every)
    env.close()


if __name__ == "__main__":
    main()
