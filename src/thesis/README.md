# Stima di V* e Q* con LLM in MiniGrid-DoorKey 8x8

Cartella `src/thesis`. Qui ho provato a capire se un LLM riesce a indovinare, senza esempi, quanto vale uno stato (`V*`) o un'azione (`Q*`) in DoorKey partendo solo da qualche mappa ASCII. Poi ho controllato se quei numeri aiutano davvero un agente a imparare più in fretta.

---

## 1. Cosa ho provato, in breve

| Cosa | A cosa serve | Dove finisce |
|---|---|---|
| `scripts/state_export2.py` | calcola il valore vero `V*` con Value Iteration e crea le storie | `files/<seed>/*.txt` e `files/metadata.csv` — 30 mappe diverse, circa 320 storie |
| `scripts/q_export.py` | stesso, ma per ogni stato aggiunge i 5 valori `Q*(s,a)` | stesse cartelle, 6 righe per stato (1 stato + 5 azioni) |
| `scripts/state_export.py` | vecchio metodo con `gamma^distanza` — tenuto solo per confronto | — |
| `llm/gconnection.py` e `llm/gconnection-q.py` | mandano le storie all'LLM con la descrizione del gioco e chiedono una stima | `result/8x8 final/V/h{1,3,5}/*.csv` e `Q/h3/v{1,2}_*.csv` |
| `result/evaluate.py` | confronta `v_value` con `v_llm`, disegna i grafici | `result/8x8 final/grafici/confronto_*.png` |
| `result/evaluate_q.py` | stessa cosa per `q_value` contro `q_llm`, più la scelta dell'azione | stessi grafici, 15 in tutto |
| `agent/doorkey_state.py` e gli agenti `doorkey_qtable*` / `doorkey_ddqn*` | provano a usare le stime LLM per dare una spinta all'addestramento | — |

Il gioco è `MiniGrid-DoorKey-8x8-v0`. Il valore vero viene sempre da Value Iteration, non da prove a caso.

---

## 2. Come leggere i numeri

Ogni riga dei CSV in `result/8x8 final/` è uno stato, o una coppia stato-azione, con il valore vero e quello dato dall'LLM.

* **MAE / RMSE** — errore medio, più basso vuol dire stima più vicina.
* **Pearson r / Spearman ρ** — quanto le due liste salgono e scendono insieme.
* **CCC** — come Pearson, ma tiene conto anche se l'LLM sbaglia di una quantità fissa o con pendenza diversa. È il numero più serio per dire “sono d’accordo?”.
* **IC 95%** — intervallo di fiducia con bootstrap sui seed. Se due intervalli non si toccano, la differenza regge.
* **bias** — media di `v_llm - v_value`. Negativo = stima troppo bassa.
* **slope / intercept** — retta `v_llm ~ v_value`. Perfetto sarebbe pendenza 1 e intercetta 0.
* **failure** — quante volte l'LLM non ha risposto (`v_llm==0`).
* **entro 0.05 / 0.10** — quante stime stanno sotto quella soglia di errore.
* **Low / Mid / High** — tre fasce per ogni tipo di storia, per vedere se l'LLM mette lo stato nella fascia giusta anche quando sbaglia il numero preciso.
* **Top-1 / 2 / 3 (solo Q)** — quante volte l'azione che l'LLM mette al primo posto è davvero la migliore. A caso sarebbe 20%.

---

## 3. Come sono fatti i CSV

### 3.1 `files/metadata.csv` — il punto di partenza

30 semi diversi, circa 11 storie ciascuno. Esempio:

```
id,seed,type,x,y,agent_dir,path,file,step_start,step_end,event,n_maps,v_value,v_llm
000001,102,initial,1,1,3,102/initial_step_0000_find_key.txt,initial_step_0000_find_key.txt,0,0,find_key,1,0.8097,0
000003,102,transition,2,4,1,102/transition_step_0005_0007_find_key.txt,transition_step_0005_0007_find_key.txt,5,7,find_key,3,0.8687,0
```

`x,y,agent_dir` dicono dove sta l’agente. `type` dice che pezzo di traiettoria è (`initial`, `worst`, `intermediate`, `transition`, `off_track`). `event` dice a che punto del compito siamo (`find_key`, `open_door`, `reach_goal`). `n_maps` è quante mappe vede l’LLM (1, 3 o 5). `v_llm` parte da 0 e poi viene riempito.

### 3.2 `result/8x8 final/V/h{1,3,5}/*.csv` — i risultati per V

```
id,seed,type,path,file,step_start,step_end,event,n_maps,v_value,v_llm
000030,860,intermediate,860/intermediate_step_0022_0024_reach_goal.txt,intermediate_step_0022_0024_reach_goal.txt,22,24,reach_goal,3,0.99,0.9801
```

Stesse colonne di prima, senza `x,y`. `v_llm==0` vuol dire risposta mancata. La cartella `h1/h3/h5` dice quante mappe c’erano nel prompt.

### 3.3 `result/8x8 final/Q/h3/v1_*.csv` — Q diretto (v1)

```
id,seed,type,path,file,step_start,step_end,event,n_maps,action,q_value,q_llm
000139,860,transition,860/transition_step_0019_0021_open_door.txt,transition_step_0019_0021_open_door.txt,19,21,open_door,3,left,0.94148,0.01
000139,860,transition,860/transition_step_0019_0021_open_door.txt,transition_step_0019_0021_open_door.txt,19,21,open_door,3,forward,0.95099,0.02
```

Stesso `id` ripetuto 5 volte, una per azione. `q_llm` diverso per ogni azione.

### 3.4 `result/8x8 final/Q/h3/v2_*.csv` — V dello stato dopo l’azione (v2)

```
id,seed,type,path,file,step_start,step_end,event,n_maps,action,q_value,q_llm
001497,413,intermediate,413/intermediate_step_0019_0021_reach_goal.txt,intermediate_step_0019_0021_reach_goal.txt,19,21,reach_goal,3,right,0.970299,0.97
```

Stessa tabella, ma `q_llm` qui è `V(s')`, cioè quanto vale lo stato in cui finisci dopo aver fatto quell’azione.

---

## 4. Prompt e cosa rispondono davvero i modelli

### 4.1 La mappa che vede l’LLM

Ogni file in `files/<seed>/*.txt` ha 1–5 mappe così, più la legenda. Esempio con due mappe, agente con chiave `L(U)` vicino al goal `G`:

```
================================================
+----+----+----+----+----+----+
|    |    | ▇  |    |    |    |
|    |    |D(O)|    |    |    |
|    |    | ▇  |    |    |    |
|    |    | ▇  |    |    |    |
|    |    | ▇  |    |    |L(U)|
|    |    | ▇  |    |    | G  |
+----+----+----+----+----+----+
================================================
+----+----+----+----+----+----+
|    |    | ▇  |    |    |    |
|    |    |D(O)|    |    |    |
|    |    | ▇  |    |    |    |
|    |    | ▇  |    |    |    |
|    |    | ▇  |    |    |L(U)|
|    |    | ▇  |    |    | G  |
```

Sotto c’è sempre: `A` = agente, `L` = con chiave, `▇` = muro, `D(L/O)` = porta, `G` = goal.

### 4.2 Prompt V — `docs/Prompt.txt`

Chiede di valutare la storia `<history_xxxx>` e di dare un solo numero `v-function-value` per ogni codice, con un campo `analisys` che spiega il ragionamento. L’output deve essere solo JSON, niente altro:

```json
[{"code": "000116", "analisys": "...", "v-function-value": 0.87}]
```

Prima delle mappe c’è la documentazione del gioco e la definizione di V.

### 4.3 Prompt Q v1 — `docs/Prompt_q.txt`

Stessa idea, ma per ogni stato chiede 5 numeri insieme:

```json
[{"code": "000116", "analisys": "...", "q-function-values": {
  "left": 0.12, "right": 0.15, "forward": 0.89, "pickup": 0.02, "drop": 0.01, "toggle": 0.01
}}]
```

### 4.4 Prompt Q v2 — `docs/en/Prompt-q2-en.txt`

Qui dentro `<history_xxxx>` c’è anche `<action>forward</action>`. L’LLM deve immaginare dove finisce dopo quell’azione e dare il V di quello stato:

```json
[{"code": "000116", "action": "forward", "analisys": "...", "v_function_value": 0.92}]
```

### 4.5 Le risposte vere, non inventate

I CSV salvano solo il numero, la parte `analisys` andava persa. Per far vedere il formato vero ho rifatto ora le stesse richieste, con stesso prompt e stessa temperatura, e ho salvato tutto in `result/real_json_examples/`. Gemma impiega circa 3 minuti a rispondere, gpt-oss pochi secondi. I numeri sono nuovi ma vicini a quelli dei CSV.

**V, gemma-4-26b, stato 000030 (h3, V vero 0.99) — 125s:**

```json
[{"code": "000030", "analisys": "L'agente ha già raccolto la chiave (L) e la porta è aperta (D(O)). L'obiettivo (G) si trova nella cella immediatamente sotto l'agente (posizione 5,5 rispetto alla 4,5). Poiché l'agente è rivolto a destra (R), l'azione ottimale è 'left' per ruotare verso il basso, seguita da 'forward' per raggiungere il traguardo. Seguendo la definizione di v-function, il ritorno atteso è $G_t = \\gamma^1 \\cdot R_{t+2}$ (poiché il reward viene erogato solo al raggiungimento del goal al secondo passo), che con $\\gamma=0.99$ e reward massimo $1.0$ è pari a $0.99$.", "v-function-value": 0.99}]
```
Nel CSV lo stesso stato aveva `v_llm` 0.9801. File: `result/real_json_examples/gemma_V_000030.json`.

**V, gpt-oss-120b, stesso stato 000030 — 2.3s:**

```json
[{"code": "000030", "analisys": "Lo stato finale mostra l'agente carico (L) a destra della porta già aperta, di fronte al goal G. L'agente può girare a destra (azione \"right\") e poi avanzare (azione \"forward\") per raggiungere il goal in due passi. Il reward per il raggiungimento del goal è quasi 1 (1 - 0.9·(step_count/max_steps)) e, con gamma=0.99, il valore atteso è ≈0.99·1 ≈ 0.99. Normalizzato nel range [0,1] il v‑function value è quindi circa 0.98.", "v-function-value": 0.98}]
```
File: `groq_V_000030.json`.

**Q v1, gemma-4-26b, stato 000030 — 157s, 5 valori insieme:**

```json
[{"code": "000030", "analisys": "L'agente si trova nello stato L(R) alla posizione (4, 5), avendo già raccolto la chiave. Il goal G è situato in (5, 5), direttamente sotto l'agente. L'azione 'right' è quella che massimizza il ritorno atteso $G_t$ secondo la definizione di $q$-function, poiché permette di orientarsi verso il basso (Down) per raggiungere il goal in soli due step ('right' seguito da 'forward'). L'azione 'left' ha un valore inferiore poiché richiede più rotazioni per l'orientamento, aumentando il numero di step $k$ e riducendo il valore scontato $\\gamma^k R$. Le azioni 'forward', 'pickup', 'drop' e 'toggle' sono inefficienti o inutili: 'forward' allontana l'agente dal goal, 'pickup' è superflua poiché la chiave è già presente, 'drop' perderebbe la chiave e 'toggle' agirebbe su una porta già aperta D(O).", "q-function-values": {"left": 0.492, "right": 0.503, "forward": 0.005, "pickup": 0.0, "drop": 0.0, "toggle": 0.0}}]
```
File: `gemma_Qv1_000030.json`.

**Q v1, gpt-oss-120b, stesso stato 000030 — 2.7s:**

```json
[{"code": "000030", "analisys": "Lo stato finale mostra l'agente caricato (L) in posizione (riga 5, colonna 6) con orientamento verso destra. La porta è già aperta (D(O)) e la casella goal (G) è subito sotto l'agente, ma l'azione forward muove nella direzione corrente (destra) ed è quindi inefficace. L'azione più vantaggiosa è girare a destra (right) per puntare verso il basso e poi avanzare (forward) per raggiungere il goal in un passo. Le altre azioni non avvicinano l'agente al goal: left lo fa girare verso l'alto, pickup è inutile perché l'agente ha già la chiave, drop è inutilizzato e toggle non ha effetto sulla porta aperta. Pertanto i valori Q sono assegnati in modo proporzionale all'utilità attesa, normalizzati fra 0 e 1.", "q-function-values": {"left": 0.1, "right": 0.9, "forward": 0.3, "pickup": 0.0, "drop": 0.0, "toggle": 0.2}}]
```
File: `groq_Qv1_000030.json`.

**Q v2, gpt-oss-120b, stato 000044 (`forward` contro muro) — 10s, singolo V(s'):**

```json
[{"code": "000044", "action": "forward", "analisys": "The forward action hits a wall, so the agent remains in the same state (position, orientation, and inventory unchanged). From this state the optimal policy must still navigate to the key, open the door, and reach the goal. Estimating the remaining optimal path length at roughly 12 steps and using γ=0.99 gives a discounted factor of about 0.99^12 ≈ 0.886. Assuming the eventual reward at the goal is close to 1, the expected return (state‑value) is approximately 0.88, which is the normalized V‑value for the resulting state.", "v_function_value": 0.88}]
```
File: `groq_Qv2_000044.json`.

---

## 5. Come ho interrogato i modelli

Ho tenuto solo due modelli: `gemma-4-26b` e `gpt-oss-120b`.

* **Contesto `h` per V:** `h1` = 1 mappa, `h3` = 3 mappe, `h5` = 5 mappe.
* **Q (h3):** `v1` contro `v2` — vedi riquadro qui sotto.
* **Senza `analisys` (solo h3, V):** stesso prompt ma senza chiedere di spiegare, per vedere quanto conta.

> **v1 contro v2 — stesso stato, richiesta diversa**
>
> |  | **v1 — Q diretto** | **v2 — V dello stato dopo** |
> |---|---|---|
> | **Cosa vede l’LLM** | 3 mappe dello stato `s` | le stesse 3 mappe più `<action>forward</action>` |
> | **Cosa deve scrivere** | 5 numeri insieme: `Q(s,left)` … `Q(s,toggle)` | un solo numero: `V(s')` dove `s'` è dove arrivi se fai quell’azione |
> | **In pratica** | `Q* = r + gamma·V(s')` — deve azzeccare tutto insieme | deve solo immaginare `s'` e dire quanto vale |
> | **Cosa cambia nei risultati** | più difficile, spesso il JSON si rompe (failure ~36%) | più semplice, quasi sempre risponde (failure 0.3–3%) e Top-1 sale a 81–90% |
>
> Detto semplice: **v1 chiede “quanto vale ogni mossa da qui?”**, **v2 chiede “quanto vale dove finisci se fai questa mossa?”**. La seconda viene meglio.

---

## 6. Risultati per V

Tutti i numeri vengono da `result/evaluate.py` su `result/8x8 final`, filtrato sui due modelli.

| Modello | n | failure | MAE [IC95%] | RMSE | Pearson r [IC95%] | CCC [IC95%] | bias | slope | entro 0.05 |
|---|---|---|---|---|---|---|---:|---:|---:|
| **h3, gemma-4-26b** | 301 | 0.0% | **0.0165 [0.0153,0.0179]** | 0.0210 | **0.922 [0.905,0.937]** | **0.915 [0.899,0.930]** | −0.0008 | 0.818 | 98.0% |
| **h5, gemma-4-26b** | 309 | 0.0% | **0.0157 [0.0137,0.0178]** | 0.0240 | 0.904 [0.872,0.931] | 0.898 [0.861,0.928] | −0.0058 | 0.859 | 95.1% |
| h1, gemma-4-26b | 319 | 0.0% | 0.0193 [0.0170,0.0214] | 0.0293 | 0.845 [0.806,0.889] | 0.836 [0.793,0.882] | −0.0028 | 0.741 | 92.5% |
| h3, gemma-4-26b senza analisys | 319 | 0.0% | 0.0292 [0.0218,0.0366] | 0.0685 | 0.611 [0.547,0.727] | 0.545 [0.444,0.711] | −0.0120 | 0.964 | 89.3% |
| h1, gpt-oss-120b | 118 | 3.3% | 0.0343 [0.0257,0.0414] | 0.0605 | 0.815 [0.778,0.868] | 0.674 [0.607,0.769] | −0.0336 | 1.236 | 83.1% |
| h5, gpt-oss-120b | 139 | 6.7% | 0.0320 [0.0207,0.0449] | 0.0863 | 0.579 [0.490,0.771] | 0.451 [0.339,0.718] | −0.0261 | 1.114 | 90.6% |
| h3, gpt-oss-120b | 123 | 1.6% | 0.0559 [0.0485,0.0634] | 0.0739 | 0.676 [0.579,0.767] | 0.558 [0.479,0.633] | −0.0265 | 1.145 | 54.5% |
| h3, gpt-oss-120b senza analisys | 203 | 9.4% | 0.0461 [0.0353,0.0596] | 0.0864 | 0.537 [0.442,0.653] | 0.443 [0.337,0.593] | −0.0189 | 0.972 | 74.4% |

### Grafici V — tutti da `result/8x8 final`

**1. Quanto sbaglia in media:**

![MAE e RMSE per modello](result/8x8%20final/grafici/confronto_00_errori.png)

Barra blu = MAE, arancione = RMSE. Segmento nero = IC 95%. Gemma h3/h5 sta quasi a zero, gpt-oss è due-tre volte più alto. Senza `analisys` l’errore di gemma raddoppia.

**2. Quanto resta fedele l’ordinamento:**

![Pearson r e CCC](result/8x8%20final/grafici/confronto_01_correlazioni.png)

Verde = Pearson, viola = CCC. Più alta la barra, meglio l’accordo. Gemma h3 tocca 0.915 di CCC, gpt-oss resta tra 0.45 e 0.67. Togliere `analisys` taglia quasi metà del CCC.

**3. Vero contro stima, un pannello per modello:**

![Scatter v_value vs v_llm](result/8x8%20final/grafici/confronto_06_scatter_riepilogo.png)

Ogni punto è uno stato. Colore = tipo di storia. Linea nera tratteggiata = perfetto (`y=x`), bande verdi = ±0.05/±0.10, linea rossa = regressione. Gemma h3/h5 è stretta attorno alla diagonale, quasi tutto entro ±0.05. gpt-oss è sparso.

**4. Quante risposte mancate:**

![Failure rate](result/8x8%20final/grafici/confronto_07_failure_rate.png)

Verde <1%, arancione <5%, rosso oltre 5%. Gemma 0% sempre. gpt-oss cresce con `h` e senza `analisys` arriva al 9.4%.

**5. Quante stime stanno sotto una soglia:**

![ECDF errori](result/8x8%20final/grafici/confronto_05_ecdf_errori.png)

Asse x = errore, y = quota di stime con errore sotto quel valore. Linee a 0.05 e 0.10. Gemma h3 a 0.05 è già al 98%, gpt-oss h3 solo al 54%.

**6. Dove sbaglia di più:**

![MAE per tipo](result/8x8%20final/grafici/confronto_03_heatmap_tipo.png)

Righe = tipo di storia, colonne = modello, numero = MAE e `n`. Giallo chiaro = poco errore, rosso = tanto. Per tutti `worst` e `initial` sono più duri, `transition` più facile.

---

## 7. Risultati per Q (h3)

### 7.1 Sul valore

| Modello | righe | failure | MAE [IC95%] | RMSE | Pearson r | CCC | bias | slope |
|---|---|---|---|---|---|---|---|---:|
| **h3, v2 gemma-4-26b** | 314 | **0.3%** | **0.1126 [0.0951,0.1349]** | 0.2078 | 0.452 | 0.180 | −0.0968 | 1.74 |
| h3, v2 gpt-oss-120b | 145 | 3.3% | 0.1548 [0.1108,0.2077] | 0.2865 | 0.332 | 0.093 | −0.1464 | 1.76 |
| h3, v1 gpt-oss-120b | 478 | 35.8% | 0.3203 [0.2817,0.3598] | 0.4471 | 0.156 | 0.025 | −0.2840 | 1.14 |
| h3, v1 gemma-4-26b | 866 | 37.5% | 0.4867 [0.4685,0.5059] | 0.5914 | 0.162 | 0.015 | −0.4753 | 1.23 |

### 7.2 Sulla scelta dell’azione

| Modello | stati | Top-1 | Top-2 | Top-3 | value loss |
|---|---|---|---|---|---|
| **h3, v2 gpt-oss-120b** | 111 | **90.1%** | 99.1% | 100% | 0.0008 |
| **h3, v2 gemma-4-26b** | 207 | 81.6% | 95.7% | 100% | 0.0016 |
| h3, v1 gemma-4-26b | 276 | 66.7% | 80.1% | 89.5% | 0.0039 |
| h3, v1 gpt-oss-120b | 141 | 61.7% | 83.0% | 92.9% | 0.0047 |

A caso sarebbe 20%.

### Grafici Q — tutti da `result/8x8 final`

**1. Errore sul valore:**

![MAE Q](result/8x8%20final/grafici/confronto_00_errori_valore.png)

v1 sta tra 0.32 e 0.48, v2 scende a 0.11–0.15. Già meglio, ma resta più alto di V.

**2. Risposte mancate:**

![Failure Q](result/8x8%20final/grafici/confronto_05_failure_rate_valore.png)

v1 perde circa un terzo dei JSON (il vettore a 5 valori si rompe spesso), v2 quasi mai.

**3. Quante volte azzecca l’azione giusta:**

![Accuracy azione](result/8x8%20final/grafici/confronto_09_accuracy_azione.png)

Anche v1 batte il caso (62–67% Top-1). Con v2 si sale a 81.6% (gemma) e 90.1% (gpt-oss), Top-2 quasi 100%.

**4. Dove è più bravo:**

![Accuracy per tipo](result/8x8%20final/grafici/confronto_10_accuracy_per_tipo.png)

`intermediate` e `transition` facili, `off_track` e `worst` duri per v1. Con v2 anche quelli fuori traccia salgono a 73–97%.

**5. Dove confonde le azioni:**

![Confusion matrix Q](result/8x8%20final/grafici/confronto_11_confusion_matrix.png)

Righe = azione davvero migliore, colonne = azione scelta dall’LLM. Diagonale = corretto. In v2 la diagonale è piena; gli errori restano quasi solo tra `left`/`right`/`forward`.

**6. Per ogni azione:**

![Accuracy per azione](result/8x8%20final/grafici/confronto_13_per_action_accuracy.png)

v1 crolla su `pickup` per gemma (28.6%) e su `left` per gpt-oss. Con v2 tutto sale sopra 74%, `pickup` di gpt-oss arriva a 100%.

---

## 8. Cosa portarsi all’incontro

* Gemma indovina V anche con una sola mappa e non sbaglia mai la risposta. gpt-oss va bene solo con poco contesto e a volte non risponde. Chiedere la spiegazione (`analisys`) aiuta entrambi.
* Per Q, chiedere direttamente `Q(s,a)` non rende. Chiedere `V(s')` dopo l’azione, invece, porta Top-1 a 80–90% e Top-2 quasi a 100%.
* Le stime di V di gemma si possono già usare per dare una spinta all’addestramento, quelle di Q solo in versione v2.

---

## 9. Struttura della cartella

```
thesis/
├── scripts/state_export2.py   # crea V* (Value Iteration)
├── scripts/q_export.py        # crea Q* / V(s')
├── llm/gconnection.py         # chiama l’LLM (V)
├── llm/gconnection-q.py       # chiama l’LLM (Q v1)
├── llm/gconnection-q2.py      # chiama l’LLM (V(s') v2)
├── docs/Prompt.txt            # prompt V
├── docs/Prompt_q.txt          # prompt Q
├── docs/en/                   # varianti in inglese e definizioni
├── files/metadata.csv         # valori veri (30 semi)
├── files/<seed>/*.txt         # le mappe ASCII
├── result/8x8 final/V/h{1,3,5}/  # CSV con le stime LLM
├── result/8x8 final/Q/h3/        # CSV Q v1/v2
├── result/evaluate.py         # fa i conti per V
├── result/evaluate_q.py       # fa i conti per Q
├── result/8x8 final/grafici/  # grafici di questa sintesi (12 immagini)
├── result/real_json_examples/ # JSON veri catturati ora (5 file)
├── agent/doorkey_state.py     # codifica dello stato, funzioni di supporto
├── agent/doorkey_qtable*.py   # prove con Q tabulare
├── agent/doorkey_ddqn*.py     # prove con DDQN
└── env/                       # wrapper di MiniGrid
```

Tutti i numeri di §6–7 vengono dai `log_valutazione.txt` creati da `result/evaluate.py` e `evaluate_q.py` su `result/8x8 final` filtrato a gemma-4-26b e gpt-oss-120b.
