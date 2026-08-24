# The Action-Value Function also know as Q-Function

## Definition

In a Markov decision process, the action-value function of a policy $\pi$, denoted $q_\pi$, assigns to each state–action pair the expected return obtained by taking that action in that state and thereafter following $\pi$. Formally,

$$
q_\pi(s, a)
\doteq
\mathbb{E}_\pi\left[
G_t
\mid
S_t = s, A_t = a
\right]
=
\mathbb{E}_\pi\left[
\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
\mid
S_t = s, A_t = a
\right],
$$

where $G_t$ is the discounted return from time $t$, $\gamma \in [0,1]$ is the discount factor, and the expectation is taken with respect to the trajectory induced by $\pi$ and the environment dynamics. The quantity $q_\pi(s,a)$ is often called the $q$-function, or $Q$-function, of $\pi$.

Whereas the state-value function $v_\pi(s)$ evaluates how good it is to be in a state, the action-value function evaluates how good it is to take a particular action in that state. The two are related by averaging over the policy:

$$
v_\pi(s)
=
\sum_{a} \pi(a \mid s) \, q_\pi(s, a).
$$

Conversely, conditioning on the immediate transition yields

$$
q_\pi(s, a)
=
\mathbb{E}_\pi\left[
R_{t+1} + \gamma v_\pi(S_{t+1})
\mid
S_t = s, A_t = a
\right].
$$

## Bellman Equation for $q_\pi$

The action-value function satisfies a recursive consistency condition known as the Bellman equation. Expanding one step of the process gives

$$
q_\pi(s, a)
=
\sum_{s', r}
p(s', r \mid s, a)
\left[
r
+
\gamma \sum_{a'} \pi(a' \mid s') \, q_\pi(s', a')
\right].
$$

Thus, the value of taking action $a$ in state $s$ equals the expected immediate reward plus the discounted expected value of the next state–action pair, where the subsequent action is chosen according to $\pi$.

## Optimal Action-Value Function

Among all policies there exists at least one optimal policy $\pi_*$. All optimal policies share the same action-value function,

$$
q_*(s, a)
\doteq
\max_{\pi} q_\pi(s, a),
$$

called the optimal action-value function. It is related to the optimal state-value function by

$$
v_*(s) = \max_{a} q_*(s, a).
$$

If $q_*$ is known, an optimal policy is obtained by acting greedily with respect to it:

$$
\pi_*(s) \in \arg\max_{a} q_*(s, a).
$$

No model of the transition dynamics is required for this choice: it suffices to compare the scalar values $q_*(s, \cdot)$ in the current state.

## Bellman Optimality Equation

The optimal action-value function obeys the Bellman optimality equation

$$
q_*(s, a)
=
\mathbb{E}\left[
R_{t+1}
+
\gamma \max_{a'} q_*(S_{t+1}, a')
\mid
S_t = s, A_t = a
\right],
$$

or, written explicitly in terms of the dynamics,

$$
q_*(s, a)
=
\sum_{s', r}
p(s', r \mid s, a)
\left[
r
+
\gamma \max_{a'} q_*(s', a')
\right].
$$

The sole structural difference from the Bellman equation for $q_\pi$ is the replacement of the expectation under $\pi$ by a maximization over the next action. After the first step, the agent is assumed to behave optimally thereafter.

## Role in Control

The practical importance of $q_*$ stems from the fact that it directly supports _action selection_. Algorithms that estimate action values—most notably temporal-difference methods such as SARSA and Q-learning—can improve a policy, or learn an optimal policy, without an explicit model of the environment. In that sense, the $q$-function is the central object of model-free control in reinforcement learning.

## Summary of Relations

| Function     | Meaning                                         | Policy after the first action |
| ------------ | ----------------------------------------------- | ----------------------------- |
| $v_\pi(s)$   | Expected return from state $s$ under $\pi$      | $\pi$                         |
| $q_\pi(s,a)$ | Expected return from $(s,a)$, then $\pi$        | $\pi$                         |
| $v_*(s)$     | Maximum achievable expected return from $s$     | optimal                       |
| $q_*(s,a)$   | Maximum achievable expected return from $(s,a)$ | optimal                       |

The identities are

$$
v_\pi(s) = \sum_a \pi(a \mid s) \, q_\pi(s,a),
\qquad
v_*(s) = \max_a q_*(s,a).
$$
