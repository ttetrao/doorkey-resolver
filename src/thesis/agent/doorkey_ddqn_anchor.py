#!/usr/bin/env python3
import argparse
import csv
import random
import sys
from pathlib import Path
from typing import cast
from collections import deque

import numpy as np
import gymnasium as gym
import minigrid
from gymnasium.spaces import Discrete

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from ExperienceReplayBuffer import ExperienceReplayBuffer, Experience
from env.view_wrapper import DoorKeyViewSystem
from doorkey_state import encode, to_vec, build_known, build_potential, plot_compare

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MIN_BUFFER_FILL = 1000


class DDQNNet(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, s):
        return self.mlp(s)


class DDQNAgent:
    def __init__(self, action_dim, lr=1e-3, gamma=0.99, eps_decay=0.999,
                 buffer_size=300000, batch_size=64, lam=1.0, known_pool=()):
        self.device = device
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.lam = lam
        self.known_pool = list(known_pool)
        self.policy = DDQNNet(action_dim).to(device)
        self.target = DDQNNet(action_dim).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = ExperienceReplayBuffer(
            batch_size, buffer_size, 0.4, np.random.RandomState())
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = eps_decay
        self.beta = 0.5
        self.beta_inc = 0.00001

    def act(self, state, eval=False):
        if not eval and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        self.policy.eval()
        with torch.no_grad():
            s = torch.as_tensor(state, device=self.device).unsqueeze(0)
            q = self.policy(s)
        if not eval:
            self.policy.train()
        return q.argmax().item()

    def update(self):
        if len(self.memory) < self.memory.batch_size:
            return 0.0, 0.0
        idxs, exps, weights = self.memory.sample(self.beta)
        s = np.array([e.state for e in exps])
        ns = np.array([e.next_state for e in exps])
        act = np.array([e.action for e in exps])
        rew = np.array([e.reward for e in exps], dtype=np.float32)
        done = np.array([e.done for e in exps], dtype=np.float32)

        s_t = torch.as_tensor(s, device=self.device)
        act_t = torch.tensor(act, dtype=torch.int64).unsqueeze(1).to(self.device)
        rew_t = torch.tensor(rew).unsqueeze(1).to(self.device)
        ns_t = torch.tensor(ns, device=self.device)
        done_t = torch.tensor(done).unsqueeze(1).to(self.device)
        w_t = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        q = self.policy(s_t).gather(1, act_t)
        with torch.no_grad():
            best_acts = self.policy(ns_t).argmax(dim=1, keepdim=True)
            nq = self.target(ns_t).gather(1, best_acts)
            target = rew_t + self.gamma * nq * (1 - done_t)

        td = torch.abs(q - target).detach().cpu().numpy()
        self.memory.update_priorities(idxs, td.flatten() + 1e-5)
        loss = (w_t * nn.functional.smooth_l1_loss(q, target, reduction='none')).mean()

        anchor_loss = torch.tensor(0.0, device=self.device)
        if self.known_pool:
            items = random.sample(self.known_pool, min(self.batch_size, len(self.known_pool)))
            a_s = torch.as_tensor(np.array([o for o, _ in items]), device=self.device)
            a_v = torch.tensor([v for _, v in items], device=self.device)
            anchor_loss = self.lam * ((self.policy(a_s).max(dim=1).values - a_v) ** 2).mean()
            loss = loss + anchor_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        self.beta = min(1.0, self.beta + self.beta_inc)
        return loss.item(), anchor_loss.item()

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def decay_eps(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(env, agent, episodes, potential, csv_seeds, max_steps=0, log_every=100):
    rewards, losses, successes, ep_lengths = [], [], [], []
    sr_buffer = deque(maxlen=100)
    count = 0
    log_hits = 0
    log_phi = 0.0
    log_aux = 0.0

    for ep in range(episodes):
        p = 1.0 - 0.8 * ep / max(episodes - 1, 1)
        seed = random.choice(csv_seeds) if csv_seeds and random.random() < p else random.randint(0, 1000)
        env.reset(seed=seed)
        st = encode(env)
        s = to_vec(st)
        ep_rew = 0.0
        ep_loss = 0.0
        steps = 0
        while max_steps == 0 or steps < max_steps:
            a = agent.act(s)
            ns, r, term, trunc, info = env.step(a)
            nst = encode(env)
            ns_vec = to_vec(nst)
            phi_s = potential(st)
            phi_n = potential(nst)
            if phi_s > 0 or phi_n > 0:
                log_hits += 1
                log_phi += phi_s + phi_n
            r += agent.gamma * phi_n - phi_s
            agent.memory.add(Experience(s, a, r, ns_vec, term))
            if count % 4 == 0 and len(agent.memory) > MIN_BUFFER_FILL:
                tot, aux = agent.update()
                ep_loss += tot
                log_aux += aux
            if count % 2500 == 0:
                agent.update_target()
            ep_rew += r
            steps += 1
            count += 1
            s, st = ns_vec, nst
            if term or trunc:
                break
        agent.decay_eps()
        is_success = 1.0 if (term and not trunc and ep_rew > 0) else 0.0
        sr_buffer.append(is_success)
        rewards.append(ep_rew)
        losses.append(ep_loss / max(steps, 1))
        successes.append(is_success)
        ep_lengths.append(steps)

        if ep % log_every == 0:
            avg_r = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            phi_avg = log_phi / (2 * log_hits) if log_hits else 0.0
            print(f"Ep {ep:5d}: reward={ep_rew:.2f} avg100={avg_r:.2f} sr={np.mean(sr_buffer):.3f} eps={agent.epsilon:.3f} buf={len(agent.memory)} v_llm={log_hits} hits φavg={phi_avg:.2f} aux={log_aux:.4f}")
            log_hits = 0
            log_phi = 0.0
            log_aux = 0.0

    return {"rewards": rewards, "losses": losses, "successes": successes, "ep_lengths": ep_lengths}


def evaluate(env, agent, episodes, seeds):
    sr = []
    for i in range(episodes):
        env.reset(seed=seeds[i])
        s = to_vec(encode(env))
        ep_rew = 0.0
        while True:
            a = agent.act(s, eval=True)
            s, r, term, trunc, _ = env.step(a)
            s = to_vec(encode(env))
            ep_rew += r
            if term or trunc:
                break
        sr.append(1.0 if term and not trunc and ep_rew > 0 else 0.0)
    return np.mean(sr)


def plot_results(hist, algo_name="DDQN"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rewards = hist["rewards"]
    window = 100
    axes[0, 0].plot(rewards, alpha=0.3, label="per ep")
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(rewards)), ma, label=f"media mobile {window}")
    axes[0, 0].set_title("Reward per episodio")
    axes[0, 0].set_xlabel("Episodio")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()

    sr = np.convolve(hist["successes"], np.ones(window)/window, mode='valid')
    axes[0, 1].plot(range(window-1, len(hist["successes"])), sr)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Success Rate (100-ep media mobile)")
    axes[0, 1].set_xlabel("Episodio")
    axes[0, 1].set_ylabel("SR")

    axes[1, 0].plot(hist["losses"])
    axes[1, 0].set_title("Loss media per episodio")
    axes[1, 0].set_xlabel("Episodio")
    axes[1, 0].set_ylabel("Loss")

    axes[1, 1].plot(hist["ep_lengths"], alpha=0.3, label="per ep")
    if len(hist["ep_lengths"]) >= window:
        ma = np.convolve(hist["ep_lengths"], np.ones(window)/window, mode='valid')
        axes[1, 1].plot(range(window-1, len(hist["ep_lengths"])), ma, label=f"media mobile {window}")
    axes[1, 1].set_title("Lunghezza episodio")
    axes[1, 1].set_xlabel("Episodio")
    axes[1, 1].set_ylabel("Steps")
    axes[1, 1].legend()

    fig.suptitle(f"Training {algo_name} - MiniGrid-DoorKey-8x8")
    plt.tight_layout()
    out = Path(__file__).parent.parent / "ddqn_anchor_training.png"
    fig.savefig(out, dpi=150)
    print(f"Plot salvato: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="DDQN con regolarizzazione vincolata del Bellman residual (anchor v_llm)")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_decay", type=float, default=0.999)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=300000)
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
    torch.manual_seed(args.seed)

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

    known_pool = [(to_vec(st), v) for st, v in known.items()]

    def make_agent(lam, pool):
        return DDQNAgent(
            action_dim=n_actions,
            lr=args.lr,
            gamma=args.gamma,
            eps_decay=args.eps_decay,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            lam=lam,
            known_pool=pool,
        )

    print(f"Device: {device}")

    if args.compare:
        eval_seeds = random.sample(range(100000), 200)

        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        agent_v = make_agent(0.0, ())
        hist_v = train(env, agent_v, args.episodes, build_potential({}), [], args.max_steps, args.log_every)
        eval_v = evaluate(env, agent_v, len(eval_seeds), eval_seeds)
        print(f"Vanilla eval SR: {eval_v:.3f}")

        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        agent_a = make_agent(args.lam, known_pool)
        pot_a = build_potential(known) if args.shaping else build_potential({})
        hist_a = train(env, agent_a, args.episodes, pot_a, csv_seeds, args.max_steps, args.log_every)
        eval_a = evaluate(env, agent_a, len(eval_seeds), eval_seeds)
        print(f"Augmented eval SR: {eval_a:.3f}")

        plot_compare(hist_v, hist_a, eval_v, eval_a, "ddqn_anchor_compare.png")
        env.close()
        return

    agent = make_agent(args.lam, known_pool)
    pot = build_potential(known) if args.shaping else build_potential({})
    print(f"Training DDQN-anchor: {args.episodes} episodes, lr={args.lr}, lam={args.lam}, shaping={args.shaping}")
    hist = train(env, agent, args.episodes, pot, csv_seeds, args.max_steps, args.log_every)
    plot_results(hist)
    env.close()


if __name__ == "__main__":
    main()
