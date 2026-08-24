from dataclasses import dataclass, replace
import random
import csv
import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from monorepo import GeminiLLM, load_api_keys

DRY_RUN = False
HISTORY_PER_REQUEST = 4
NUM_REQUESTS = 1000
RETRY_LIMIT = 3
RETRY_DELAY = 10

# La tua idea: lancia una richiesta ogni 80 secondi
LAUNCH_INTERVAL_SEC = 65
# Max richieste contemporanee (180s di latency / 80s interval ≈ 2.25, teniamo 5 per sicurezza)
MAX_INFLIGHT = 5

MODEL_ID = "gemma-4-26b-a4b-it"
OUT_FILENAME = f"output_q_{MODEL_ID}.csv"

# NOTA: Hai detto "voglio ignorare solo toggle" alla domanda su "drop".
# Presumo sia un refuso e intenda ignorare "drop" (azione non presente in q_export.py).
# Se si vuole ignorare "toggle", aggiungerlo alla lista qui sotto.
IGNORED_ACTIONS = ["drop"]


@dataclass(frozen=True)
class StateEntry:
    """Rappresenta la riga (s) con entry_type=0 letta dal CSV."""

    id: str
    seed: int
    type: str
    path: str
    file: str
    step_start: int
    step_end: int
    event: str
    n_maps: int


@dataclass(frozen=True)
class QEntry:
    """Rappresenta la tupla (s, a) risultante dal LLM, pronta per il CSV di output."""

    id: str
    seed: int
    type: str
    path: str
    file: str
    step_start: int
    step_end: int
    event: str
    n_maps: int
    action: str
    q_value: float
    q_llm: float


def parse_state_entry(raw: dict) -> StateEntry | None:
    try:
        return StateEntry(
            id=raw["id"],
            seed=int(raw["seed"]),
            type=raw["type"],
            path=raw["path"],
            file=raw["file"],
            step_start=int(raw["step_start"]),
            step_end=int(raw["step_end"]),
            event=raw["event"],
            n_maps=int(raw["n_maps"]),
        )
    except (KeyError, ValueError) as e:
        print(f"  errore parsing riga CSV: {e} — {raw.get('id', '?')}")
        return None


_file_cache: dict[str, str] = {}
_file_cache_lock = threading.Lock()


def read_history(base: str, path: str) -> str | None:
    full = os.path.join(base, "files", path)
    if full not in _file_cache:
        with _file_cache_lock:
            if full not in _file_cache:
                try:
                    _file_cache[full] = open(full).read()
                except FileNotFoundError:
                    print(f"  file non trovato: {full}")
                    _file_cache[full] = None
    return _file_cache[full]


def call_with_retry(
    client, prompt, retries=RETRY_LIMIT, delay=RETRY_DELAY
) -> str | None:
    for attempt in range(retries):
        try:
            response = client.ask(prompt=prompt)
            print(f"raw: {response}")
            if response and response.strip():
                return response
            print(f"  risposta vuota (tentativo {attempt + 1}/{retries})")
        except Exception as e:
            print(f"  errore richiesta (tentativo {attempt + 1}/{retries}): {e}")
        if attempt < retries - 1:
            time.sleep(delay)
    return None


def load_documentation(base: str) -> tuple[str, str]:
    doc = open(os.path.join(base, "docs/MiniGridDocumentation2.md")).read()
    vfun_def = open(os.path.join(base, "docs/en/q-function_definition.md")).read()
    return doc, vfun_def


def build_base_prompt(base: str, doc: str, vfun_def: str) -> str:
    # Carichiamo il nuovo prompt per la Q-function
    tmpl = open(os.path.join(base, "docs/en/Prompt-q-en.txt")).read()
    return tmpl.replace(
        "X<legend>",
        f"<documentation>\n{doc}\n</documentation>\n<definition>\n{vfun_def}\n</definition>\n<legend>",
    )


def build_history_prompt(base_prompt: str, base: str, sample: list[StateEntry]) -> str:
    parts = [base_prompt]
    for h in sample:
        content = read_history(base, h.path)
        if content is None:
            content = f"(file non trovato: {h.path})"
        parts.append(f"\n<history_{h.id}>\n{content}\n</history_{h.id}>")
    return "".join(parts)


_save_lock = threading.Lock()


def save_output(path: str, entries: list[QEntry]):
    fieldnames = list(QEntry.__dataclass_fields__.keys())
    with _save_lock:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in entries:
                writer.writerow(e.__dict__)


def parse_response(
    response, batch: list[StateEntry], real_q_lookup: dict
) -> list[QEntry]:
    try:
        results = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\[\s*\{.*\}\s*\]", response, re.DOTALL)
        if not match:
            return []
        try:
            results = json.loads(match.group())
        except json.JSONDecodeError:
            return []

    if not isinstance(results, list):
        return []

    out = []
    for item in results:
        code = item.get("code", "").removeprefix("history_")
        match_state = next((e for e in batch if e.id == code), None)
        if match_state is None:
            continue

        q_funcs = item.get("q-function", [])
        if not isinstance(q_funcs, list):
            continue

        for q_obj in q_funcs:
            action = q_obj.get("action", "").lower().strip()
            if not action or action in IGNORED_ACTIONS:
                continue

            try:
                q_llm = float(q_obj.get("q-value", 0.0))
            except (ValueError, TypeError):
                continue

            # Recupero la q_value reale calcolata da value iteration tramite la chiave
            key = (
                str(match_state.seed),
                match_state.type,
                str(match_state.step_start),
                str(match_state.step_end),
                action,
            )
            q_real = real_q_lookup.get(key, 0.0)

            out.append(
                QEntry(
                    id=match_state.id,
                    seed=match_state.seed,
                    type=match_state.type,
                    path=match_state.path,
                    file=match_state.file,
                    step_start=match_state.step_start,
                    step_end=match_state.step_end,
                    event=match_state.event,
                    n_maps=match_state.n_maps,
                    action=action,
                    q_value=q_real,
                    q_llm=q_llm,
                )
            )
    return out


def process_batch(client, base_prompt, base, batch, batch_idx, total, real_q_lookup):
    prompt = build_history_prompt(base_prompt, base, batch)
    ids = ", ".join(e.id for e in batch)
    print(f"[Batch {batch_idx}/{total}] Lanciato ({ids})...")

    t0 = time.time()
    response = call_with_retry(client, prompt)
    elapsed = time.time() - t0
    print(f"[Batch {batch_idx}/{total}] Risposta in {elapsed:.1f}s")

    if response is None:
        return []

    return parse_response(response, batch, real_q_lookup)


def run():
    load_api_keys()
    client = GeminiLLM(model_id=MODEL_ID, temperature=0.7)

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc, vfun_def = load_documentation(base)
    base_prompt = build_base_prompt(base, doc, vfun_def)

    states: list[StateEntry] = []
    real_q_lookup = {}

    # Leggiamo il metadata.csv e dividiamo gli stati (entry_type=0) dalle q_values realli (entry_type=1)
    with open(os.path.join(base, "files/metadata.csv")) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["entry_type"] == "0":
                state = parse_state_entry(r)
                if state:
                    states.append(state)
            elif r["entry_type"] == "1":
                # Creiamo una lookup table per recuperare la Q-value ottima
                key = (
                    r["seed"],
                    r["type"],
                    r["step_start"],
                    r["step_end"],
                    r["action"],
                )
                try:
                    real_q_lookup[key] = float(r["q_value"])
                except ValueError:
                    pass

    out_path = os.path.join(base, "files", OUT_FILENAME)

    # --- LOGICA DI RIPRESA ---
    existing_ids = set()
    output: list[QEntry] = []
    if os.path.exists(out_path):
        print(f"Trovato output esistente: {out_path}")
        with open(out_path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    p = QEntry(
                        id=r["id"],
                        seed=int(r["seed"]),
                        type=r["type"],
                        path=r["path"],
                        file=r["file"],
                        step_start=int(r["step_start"]),
                        step_end=int(r["step_end"]),
                        event=r["event"],
                        n_maps=int(r["n_maps"]),
                        action=r["action"],
                        q_value=float(r["q_value"]),
                        q_llm=float(r["q_llm"]),
                    )
                    existing_ids.add(p.id)
                    output.append(p)
                except (KeyError, ValueError):
                    continue
        original = len(states)
        states = [e for e in states if e.id not in existing_ids]
        print(f"Saltate {original - len(states)} entry (stati) già processate.")
        if not states:
            print("Tutte le entry già processate. Uscita.")
            return

    random.shuffle(states)
    states = states[: HISTORY_PER_REQUEST * NUM_REQUESTS]

    if DRY_RUN:
        print("DRY_RUN=True — nessuna richiesta API.")
        return

    batches = [
        states[i : i + HISTORY_PER_REQUEST]
        for i in range(0, len(states), HISTORY_PER_REQUEST)
    ]
    total = len(batches)

    print(f"Avvio {total} batch. Pacing: 1 batch ogni {LAUNCH_INTERVAL_SEC}s.")

    def on_done(future):
        try:
            results = future.result()
            for entry in results:
                output.append(entry)
            save_output(out_path, output)
            print(f"  + Salvato. Totale entry accumulate: {len(output)}")
        except Exception as e:
            print(f"  Errore nel thread: {e}")

    with ThreadPoolExecutor(max_workers=MAX_INFLIGHT) as pool:
        futures = []
        for batch_idx, batch in enumerate(batches, start=1):
            fut = pool.submit(
                process_batch,
                client,
                base_prompt,
                base,
                batch,
                batch_idx,
                total,
                real_q_lookup,
            )
            fut.add_done_callback(on_done)
            futures.append(fut)

            if batch_idx < total:
                time.sleep(LAUNCH_INTERVAL_SEC)

        print(
            "\nTutti i batch sono stati lanciati. Attendo il completamento delle risposte in corso..."
        )
        for fut in futures:
            fut.result()

    print(f"\nFatto. {len(output)} entry totali in {out_path}")


if __name__ == "__main__":
    run()
