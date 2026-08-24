## Definition of the \(q\)-function

In the book _Reinforcement Learning: An Introduction_ by Richard S. Sutton and Andrew G. Barto (second edition), the **action-value function** (or \(q\)-function) for policy \(\pi\), denoted \(q\_\pi(s, a)\), is the expected return starting from state \(s\), taking action \(a\), and following policy \(\pi\) thereafter.

Formally, for Markov Decision Processes (MDPs):

\[
q*\pi(s, a) \doteq \mathbb{E}*\pi[G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R\_{t+k+1} \Bigm| S_t = s, A_t = a\right]
\]

where:

- \(G_t\) is the return starting from time \(t\)
- \(\gamma \in [0,1]\) is the discount factor
- \(\mathbb{E}\_\pi[\cdot]\) denotes the expected value under policy \(\pi\)
- \(R\_{t+k+1}\) is the reward received at time \(t+k+1\)
- \(\mathcal{S}\) is the state space and \(\mathcal{A}\) is the action space

## Intuitive meaning

The \(q\)-function specifically evaluates the value of a _particular action_ taken in that state. It indicates the total reward an agent can expect to accumulate by starting its transition with action \(a\) from state \(s\) and subsequently following policy \(\pi\).

## Optimal value function

There always exists at least one optimal policy \(\pi\_\*\). All optimal policies share the same optimal action-value function:

\[
q*\*(s, a) \doteq \max*\pi q\_\pi(s, a)
\]

This function provides the maximum expected reward obtainable starting from \(s\) and executing \(a\), then following the best possible policy thereafter. If \(q\_\*(s, a)\) is known, the optimal choice simply consists of choosing, in each state \(s\), the action \(a\) that maximizes the \(Q\)-value.

## Bellman equation for the \(q\)-function

The \(q\)-function satisfies the Bellman equation. In particular, for the optimal \(q\_\*\):

\[
q*\*(s, a) = \mathbb{E}[R*{t+1} + \gamma \max*{a'} q*\*(S\_{t+1}, a') \mid S_t = s, A_t = a]
\]

The value of taking an action in a state equals the expected immediate reward plus the maximum discounted value achievable from the next state.
