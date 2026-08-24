## Definizione di \(q\)-function

Nel libro *Reinforcement Learning: An Introduction* di Richard S. Sutton e Andrew G. Barto (seconda edizione), la **action-value function** (o \(q\)-function) per la policy \(\pi\), denotata \(q_\pi(s, a)\), è il ritorno atteso partendo dallo stato \(s\), compiendo l'azione \(a\), e seguendo la policy \(\pi\) da lì in poi.

Formalmente, per i processi decisionali di Markov (MDP):

\[
q_\pi(s, a) \doteq \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \Bigm| S_t = s, A_t = a\right]
\]

dove:
- \(G_t\) è il ritorno (return) a partire dal tempo \(t\)
- \(\gamma \in [0,1]\) è il fattore di sconto
- \(\mathbb{E}_\pi[\cdot]\) indica il valore atteso sotto la policy \(\pi\)
- \(R_{t+k+1}\) è la ricompensa ricevuta al tempo \(t+k+1\)
- \(\mathcal{S}\) è lo spazio degli stati e \(\mathcal{A}\) lo spazio delle azioni

## Significato intuitivo

Mentre la \(v\)-function misura quanto è "buono" uno stato in generale, la \(q\)-function valuta specificamente il valore di una *particolare azione* presa in quello stato. Indica il totale di reward che un agente può aspettarsi di accumulare iniziando la sua transizione con l'azione \(a\) dallo stato \(s\) e seguendo successivamente la policy \(\pi\).

## Value function ottimale

Esiste sempre almeno una policy ottimale \(\pi_*\). Tutte le policy ottimali condividono la stessa action-value function ottimale:

\[
q_*(s, a) \doteq \max_\pi q_\pi(s, a)
\]

Questa funzione fornisce la massima ricompensa attesa ottenibile partendo da \(s\) ed eseguendo \(a\), seguendo poi la policy migliore possibile. Se si conosce la \(q_*(s, a)\), la scelta ottima consiste semplicemente nello scegliere, in ogni stato \(s\), l'azione \(a\) che massimizza il \(Q\)-value.

## Equazione di Bellman per la \(q\)-function

Come la \(v\)-function, anche la \(q\)-function soddisfa l'equazione di Bellman. In particolare, per l'ottimale \(q_*\):

\[
q_*(s, a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \mid S_t = s, A_t = a]
\]

Il valore di compiere un'azione in uno stato è uguale al reward immediato atteso più il massimo valore scontato raggiungibile dallo stato successivo.
