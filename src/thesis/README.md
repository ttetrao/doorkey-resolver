# Stima di V* e Q* con LLM in MiniGrid-DoorKey 8×8 — Sintesi per l'incontro

> Cartella `src/thesis`. Obiettivo: verificare se un LLM riesce a stimare zero-shot la funzione valore ottima `V*(s)` e `Q*(s,a)` in MiniGrid-DoorKey a partire da una storia di osservazioni ASCII, e se tali stime sono utilizzabili per accelerare l'apprendimento per rinforzo.

---

## 1. Cosa è stato provato

| Script | Cosa fa | Output |
|---|---|---|
| `scripts/state_export2.py` | Calcola `V*` ottima con Value Iteration (`γ=0.99`) sull'MDP deterministico DoorKey 8×8 e genera il dataset di storie. Ogni seed produce 5 tipi di traiettoria: `initial`, `worst`, `intermediate`, `transition`, `off_track`, con `n_maps` = numero di mappe ASCII mostrate all'LLM. | `files/<seed>/*.txt` + `files/metadata.csv` (30 seed, ~320 righe) |
| `scripts/q_export.py` | Come sopra, ma emette per ogni stato anche le 5 `Q*(s,a)` con `entry_type`. | Stessa struttura, 6× righe (1 stato + 5 azioni) |
| `scripts/state_export.py` | Versione precedente con euristica `γ^distanza` via BFS — deprecata. | — |
| `llm/gconnection.py`, `llm/gconnection-q.py` | Interroga l'LLM con `docs/Prompt.txt` (V) o `docs/Prompt_q.txt` (Q). Prompt = documentazione + definizione V/Q + legenda + 1–5 storie ASCII. | `result/8x8 final/V/h{1,3,5}/*.csv` e `Q/h3/v{1,2}_*.csv` |
| `result/evaluate.py` | Valuta `v_value` vs `v_llm` e genera 9 grafici di confronto globale + 8 per file. | `grafici/confronto_*.png` + `log_valutazione.txt` |
| `result/evaluate_q.py` | Valuta `q_value` vs `q_llm` + selezione azione (Top-1/2/3, value loss, confusion matrix). | Stesso schema per Q |
| `agent/doorkey_state.py` + `agent/doorkey_qtable.py`, `doorkey_qtable_anchor.py`, `doorkey_qtable_warmstart.py`, `doorkey_ddqn.py`, `doorkey_ddqn_anchor.py` | Testano l'uso delle stime LLM in RL tabulare e DDQN tramite reward shaping potenziale, anchor sul Bellman e warm-start di Q. | `qtable_compare.png`, `ddqn_compare.png`, ecc. |

**Ambiente:** `MiniGrid-DoorKey-8x8-v0` (wrapper in `env/` per la vista ASCII e gli eventi chiave/porta/goal). Ground truth sempre da Value Iteration.

---

## 2. Legenda — cosa contengono i CSV e cosa misurano gli script di valutazione

### Contenuto dei CSV in `result/8x8 final/`

Ogni riga è uno stato (V) o una coppia stato-azione (Q) con:

* `v_value` / `q_value` — valore ottimo di riferimento, calcolato con Value Iteration (`γ=0.99`, reward `1 − 0.9·step/max_steps`). È il ground truth.
* `v_llm` / `q_llm` — stima prodotta dall'LLM per quello stesso stato/azione, parsata dal JSON di risposta. Se `== 0` l'LLM non ha risposto ed è contata come **failure**.
* `type` — tipologia della traiettoria (`initial`, `worst`, `intermediate`, `transition`, `off_track`).
* `event` — fase del task al momento dello stato (`find_key`, `open_door`, `reach_goal`).
* `n_maps` — quante mappe ASCII di contesto sono state mostrate nel prompt (`h1=1`, `h3=3`, `h5=5`).
* `seed` — configurazione iniziale della mappa (30 seed diversi).

I grafici e le tabelle sotto non riportano i CSV grezzi, ma il confronto `v_value` vs `v_llm` (o `q_value` vs `q_llm`).

### Cosa misurano `evaluate.py` e `evaluate_q.py`

* **MAE / RMSE** — errore medio assoluto e quadratico tra valore vero e stima LLM (più basso = meglio).
* **Pearson r / Spearman ρ** — correlazione lineare e di rango (quanto le due serie variano insieme, −1..1).
* **CCC (Lin)** — concordanza: come Pearson ma penalizza anche bias sistematico e slope ≠ 1 (metrica di accordo principale).
* **IC 95%** — intervallo di confidenza al 95% via bootstrap sui seed (se gli intervalli di due modelli non si sovrappongono, la differenza è significativa).
* **bias** — media di `v_llm − v_value` (negativo = sottostima, positivo = sovrastima).
* **slope / intercept** — retta di regressione `v_llm ~ v_value` (ideale: slope 1, intercept 0).
* **failure rate** — quota di righe con `v_llm/q_llm == 0` (LLM non ha risposto).
* **entro 0.05 / 0.10** — percentuale di stime con errore sotto soglia.
* **Accordo categoriale Low/Mid/High** — soglie ai quantili 33°/66° per ogni `type`: verifica se LLM e ground truth cadono nella stessa fascia (utile quando il valore assoluto è meno importante dell'ordinamento).
* **Top-1/2/3 accuracy (solo Q)** — quante volte l'azione con `q_llm` più alta coincide con quella ottima (`argmax q_value`). Baseline casuale 20% (5 azioni). **Value loss** = perdita di reward per aver scelto l'azione LLM invece dell'ottima.

---

## 3. Interrogazione LLM — configurazioni valutate qui

Questa sintesi riporta **solo** i due modelli richiesti:

* **Modelli:** `gemma-4-26b` e `gpt-oss-120b`
* **Contesto `h` (V):** `h1` (1 mappa), `h3` (3 mappe), `h5` (5 mappe)
* **Formulazione Q (h3):** `v1` vs `v2` (vedi riquadro sotto)
* **Variante `no_analisys` (solo h3, V):** stesso prompt senza campo `analisys` — misura l'effetto del ragionamento esplicito richiesto

> **Differenza v1 vs v2 — stesso stato `s`, compito diverso per l'LLM**
>
> |  | **v1 — stima diretta di `Q(s,a)`** | **v2 — stima di `V(s')`** |
> |---|---|---|
> | **Cosa riceve l'LLM** | Storia di 3 mappe ASCII dello stato `s` | Stessa storia di `s` **+ un tag `<action>`** che indica una singola azione da valutare (es. `<action>forward</action>`) |
> | **Cosa deve produrre** | Un vettore di 5 valori in un colpo solo: `Q(s,left)`, `Q(s,right)`, `Q(s,forward)`, `Q(s,pickup)`, `Q(s,toggle)` (`docs/Prompt_q.txt`, JSON con 5 campi `q-value`) | Un singolo valore: `V(s')` dove `s'` è lo stato che si raggiungerebbe eseguendo quell'azione `a` in `s` (`docs/en/Prompt-q2-en.txt`, JSON con 1 campo `v_function_value` + `action`) |
> | **Matematicamente** | `Q*(s,a)` = `r(s,a) + γ·V*(s')` — l'LLM deve stimare insieme reward immediato, dinamica e valore futuro | `V*(s')` — l'LLM immagina solo lo stato successore e ne valuta la bontà; il `r` è implicito nel valore di `s'` |
> | **Perché i risultati cambiano** | Più difficile: 5 stime correlate, l'LLM confonde azioni simili e omette spesso il JSON (failure ~36%). | Più semplice: 1 stima alla volta, decompone il problema in "cosa fa questa azione?" + "quanto vale lo stato risultante". Failure crolla a 0.3–3.3%, MAE dimezzato e Top-1 sale a 81–90% (vedi §5). |
>
> In breve: **v1 chiede "quanto vale ogni mossa da qui?"**, **v2 chiede "quanto vale lo stato in cui finisci se fai questa mossa?"**. Il secondo è lo stesso giudizio di valore usato per `V*`, quindi riesce meglio.

Tutti i numeri sotto provengono dagli script di `result/`, filtrati sui file indicati.

---

## 4. Risultati — stima di V*

### Tabella globale (8 file)

| Modello / config | n validi | failure | MAE [IC95%] | RMSE | Pearson r [IC95%] | CCC [IC95%] | bias | slope | entro 0.05 |
|---|---|---|---|---|---|---|---:|---:|---:|
| **h3 · gemma-4-26b** | 301 | 0.0% | **0.0165 [0.0153,0.0179]** | 0.0210 | **0.922 [0.905,0.937]** | **0.915 [0.899,0.930]** | −0.0008 | 0.818 | 98.0% |
| **h5 · gemma-4-26b** | 309 | 0.0% | **0.0157 [0.0137,0.0178]** | 0.0240 | 0.904 [0.872,0.931] | 0.898 [0.861,0.928] | −0.0058 | 0.859 | 95.1% |
| h1 · gemma-4-26b | 319 | 0.0% | 0.0193 [0.0170,0.0214] | 0.0293 | 0.845 [0.806,0.889] | 0.836 [0.793,0.882] | −0.0028 | 0.741 | 92.5% |
| h3 · gemma-4-26b_no_analisys | 319 | 0.0% | 0.0292 [0.0218,0.0366] | 0.0685 | 0.611 [0.547,0.727] | 0.545 [0.444,0.711] | −0.0120 | 0.964 | 89.3% |
| h1 · gpt-oss_120b | 118 | 3.3% | 0.0343 [0.0257,0.0414] | 0.0605 | 0.815 [0.778,0.868] | 0.674 [0.607,0.769] | −0.0336 | 1.236 | 83.1% |
| h5 · gpt-oss_120b | 139 | 6.7% | 0.0320 [0.0207,0.0449] | 0.0863 | 0.579 [0.490,0.771] | 0.451 [0.339,0.718] | −0.0261 | 1.114 | 90.6% |
| h3 · gpt-oss_120b | 123 | 1.6% | 0.0559 [0.0485,0.0634] | 0.0739 | 0.676 [0.579,0.767] | 0.558 [0.479,0.633] | −0.0265 | 1.145 | 54.5% |
| h3 · gpt-oss_120b_no_analisys | 203 | 9.4% | 0.0461 [0.0353,0.0596] | 0.0864 | 0.537 [0.442,0.653] | 0.443 [0.337,0.593] | −0.0189 | 0.972 | 74.4% |

Giudizio automatico dello script: miglior CCC = `h3 · gemma-4-26b` (0.915). I due file con `failure ≥5%` sono esclusi dal giudizio.

---

## 5. Risultati — stima di Q* (h3)

> Le due righe `v1` e `v2` sotto corrispondono alle due formulazioni appena descritte: `v1` stima 5 `Q(s,a)` insieme, `v2` stima un singolo `V(s')` per ogni coppia `(s,a)`.

### 5.1 Valore

| Modello / config | q_rows | failure | MAE [IC95%] | RMSE | Pearson r | CCC | bias | slope |
|---|---|---|---|---|---|---|---|---:|
| **h3 · v2_gemma-4-26b** | 314 | **0.3%** | **0.1126 [0.0951,0.1349]** | 0.2078 | 0.452 | 0.180 | −0.0968 | 1.74 |
| h3 · v2_gpt-oss_120b | 145 | 3.3% | 0.1548 [0.1108,0.2077] | 0.2865 | 0.332 | 0.093 | −0.1464 | 1.76 |
| h3 · v1_gpt-oss_120b | 478 | 35.8% | 0.3203 [0.2817,0.3598] | 0.4471 | 0.156 | 0.025 | −0.2840 | 1.14 |
| h3 · v1_gemma-4-26b | 866 | 37.5% | 0.4867 [0.4685,0.5059] | 0.5914 | 0.162 | 0.015 | −0.4753 | 1.23 |

### 5.2 Selezione azione (l'LLM sceglie l'azione con Q più alta)

| Modello / config | stati validi | Top-1 | Top-2 | Top-3 | value loss | CCC valore |
|---|---|---|---|---|---|---|
| **h3 · v2_gpt-oss_120b** | 111 | **90.1%** | 99.1% | 100% | 0.0008 | 0.093 |
| **h3 · v2_gemma-4-26b** | 207 | 81.6% | 95.7% | 100% | 0.0016 | 0.180 |
| h3 · v1_gemma-4-26b | 276 | 66.7% | 80.1% | 89.5% | 0.0039 | 0.015 |
| h3 · v1_gpt-oss_120b | 141 | 61.7% | 83.0% | 92.9% | 0.0047 | 0.025 |

Baseline casuale = 20% (5 azioni). Grafici di riferimento: `confronto_10_accuracy_per_tipo.png`, `confronto_11_confusion_matrix.png`.

---

## 6. Lettura complessiva

* **V* è stimato con alta fedeltà da gemma-4-26b** anche con poco contesto (h1) e senza failure; gpt-oss-120b è utilizzabile solo a h1 e con cautela. Il campo `analisys` è necessario per entrambi.
* **Q* va chiesto come V(s'), non come Q(s,a).** Così entrambi i modelli diventano affidabili per la scelta dell'azione (Top-2 ≈ 100%).
* **Implicazione per RL:** le stime V di gemma sono già pronte per shaping potenziale, quelle Q vanno usate solo in versione v2. I 4 confronti RL (`qtable_compare.png`, `qtable_anchor_compare.png`, `qtable_warmstart_compare.png`, `ddqn_compare.png`) mostrano l'effetto di shaping/anchor/warm-start su 3000 episodi.

---

## 7. Struttura della cartella

```
thesis/
├── scripts/state_export2.py   # dataset V* (Value Iteration)
├── scripts/q_export.py        # dataset Q* / V(s')
├── llm/gconnection.py         # query LLM (V)
├── llm/gconnection-q.py       # query LLM (Q v1)
├── llm/gconnection-q2.py      # query LLM (V(s') v2)
├── docs/Prompt.txt            # prompt V (IT)
├── docs/Prompt_q.txt          # prompt Q (IT)
├── docs/en/                   # varianti EN + definizioni V/Q
├── files/metadata.csv         # ground truth V*/Q* (30 seed)
├── files/<seed>/*.txt         # storie ASCII per seed
├── result/8x8 final/V/h{1,3,5}/  # CSV LLM già pronti
├── result/8x8 final/Q/h3/        # CSV Q v1/v2
├── result/evaluate.py         # valutazione V
├── result/evaluate_q.py       # valutazione Q
├── agent/doorkey_state.py     # encode, build_known, build_potential, plot_compare
├── agent/doorkey_qtable*.py   # RL tabulare (shaping / anchor / warmstart)
├── agent/doorkey_ddqn*.py     # DDQN + shaping/anchor
└── env/                       # wrapper MiniGrid DoorKey
```

---

## 8. Riprodurre i numeri di questa sintesi

```bash
# 1. (opzionale) rigenera il dataset — 30 seed, 8×8, h=5
python thesis/scripts/state_export2.py --size 8 --history 5
python thesis/scripts/q_export.py --size 8 --history 5

# 2. interroga l'LLM (riprende da CSV esistente se già presente)
python thesis/llm/gconnection.py        # V, h=3 di default
python thesis/llm/gconnection-q.py     # Q v1
python thesis/llm/gconnection-q2.py    # Q v2 / V(s')

# 3. valuta SOLO gemma-4-26b + gpt-oss-120b (come in questa sintesi)
mkdir -p /tmp/filtered_V_all/h1 /tmp/filtered_V_all/h3 /tmp/filtered_V_all/h5
cp thesis/result/8x8\ final/V/h1/gemma-4-26b.csv thesis/result/8x8\ final/V/h1/gpt-oss_120b.csv /tmp/filtered_V_all/h1/
cp thesis/result/8x8\ final/V/h3/gemma-4-26b.csv thesis/result/8x8\ final/V/h3/gemma-4-26b_no_analisys.csv thesis/result/8x8\ final/V/h3/gpt-oss_120b.csv thesis/result/8x8\ final/V/h3/gpt-oss_120b_no_analisys.csv /tmp/filtered_V_all/h3/
cp thesis/result/8x8\ final/V/h5/gemma-4-26b.csv thesis/result/8x8\ final/V/h5/gpt-oss_120b.csv /tmp/filtered_V_all/h5/
python thesis/result/evaluate.py --path /tmp/filtered_V_all --outdir /tmp/eval_V_all
# log: /tmp/eval_V_all/log_valutazione.txt  grafici: /tmp/eval_V_all/grafici/confronto_*.png

mkdir -p /tmp/filtered_Q_all/h3
cp thesis/result/8x8\ final/Q/h3/v1_gemma-4-26b.csv thesis/result/8x8\ final/Q/h3/v1_gpt-oss_120b.csv thesis/result/8x8\ final/Q/h3/v2_gemma-4-26b.csv thesis/result/8x8\ final/Q/h3/v2_gpt-oss_120b.csv /tmp/filtered_Q_all/h3/
python thesis/result/evaluate_q.py --path /tmp/filtered_Q_all --outdir /tmp/eval_Q_all
# log: /tmp/eval_Q_all/log_valutazione.txt
```

Tutti i blocchi di numeri in §4–5 sono estratti verbatim da quei due `log_valutazione.txt` (bootstrap 500 sui seed, CCC di Lin, IC 95%).
