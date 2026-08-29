from dataclasses import dataclass, replace
import random
import csv
import json
import os
import re
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
HISTORY_PER_REQUEST = 6
NUM_REQUESTS = 100
RETRY_LIMIT = 3
RETRY_DELAY = 10

# La tua idea: lancia una richiesta ogni 80 secondi
LAUNCH_INTERVAL_SEC = 65
# Max richieste contemporanee (180s di latency / 80s interval ≈ 2.25, teniamo 5 per sicurezza)
MAX_INFLIGHT = 5

MODEL_ID = "gemma-4-26b-a4b-it"


@dataclass(frozen=True)
class Entry:
    id: str
    seed: int
    type: str
    path: str
    file: str
    step_start: int
    step_end: int
    event: str
    n_maps: int
    v_value: float
    v_llm: float


def parse_entry(raw: dict) -> Entry | None:
    try:
        return Entry(
            id=raw["id"],
            seed=int(raw["seed"]),
            type=raw["type"],
            path=raw["path"],
            file=raw["file"],
            step_start=int(raw["step_start"]),
            step_end=int(raw["step_end"]),
            event=raw["event"],
            n_maps=int(raw["n_maps"]),
            v_value=float(raw["v_value"]),
            v_llm=float(raw["v_llm"]),
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
            if response and response.strip():
                return response
            print(f"  risposta vuota (tentativo {attempt + 1}/{retries})")
        except Exception as e:
            print(f"  errore richiesta (tentativo {attempt + 1}/{retries}): {e}")
        if attempt < retries - 1:
            time.sleep(delay)
    return None


def load_documentation(base: str) -> tuple[str, str]:
    doc = open(os.path.join(base, "docs/MiniGridDocumentation.md")).read()
    vfun_def = open(os.path.join(base, "docs/definizione v-function.md")).read()
    return doc, vfun_def


def build_base_prompt(base: str, doc: str, vfun_def: str) -> str:
    tmpl = open(os.path.join(base, "docs/Prompt.txt")).read()
    return tmpl.replace(
        "X<legend>",
        f"<documentation>\n{doc}\n</documentation>\n<definition>\n{vfun_def}\n</definition>\n<legend>",
    )


def build_history_prompt(base_prompt: str, base: str, sample: list[Entry]) -> str:
    parts = [base_prompt]
    for h in sample:
        content = read_history(base, h.path)
        if content is None:
            content = f"(file non trovato: {h.path})"
        parts.append(f"\n<history_{h.id}>\n{content}\n</history_{h.id}>")
    return "".join(parts)


_save_lock = threading.Lock()


def save_output(path: str, entries: list[Entry]):
    fieldnames = list(Entry.__dataclass_fields__.keys())
    with _save_lock:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in entries:
                writer.writerow(e.__dict__)


def parse_response(response, batch):
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
        match = next((e for e in batch if e.id == code), None)
        if match is None:
            continue
        try:
            v = float(item["v-function-value"])
            out.append((match, v))
        except (KeyError, ValueError):
            continue
    return out


def process_batch(client, base_prompt, base, batch, batch_idx, total):
    prompt = build_history_prompt(base_prompt, base, batch)
    ids = ", ".join(e.id for e in batch)
    print(f"[Batch {batch_idx}/{total}] Lanciato ({ids})...")

    t0 = time.time()
    response = call_with_retry(client, prompt)
    print(response)
    elapsed = time.time() - t0
    print(f"[Batch {batch_idx}/{total}] Risposta in {elapsed:.1f}s")

    if response is None:
        return []

    return parse_response(response, batch)


def run():
    load_api_keys()
    client = GeminiLLM(model_id=MODEL_ID, temperature=0.7)

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc, vfun_def = load_documentation(base)
    base_prompt = build_base_prompt(base, doc, vfun_def)

    with open(os.path.join(base, "files/metadata.csv")) as f:
        entries = [p for r in csv.DictReader(f) if (p := parse_entry(r)) is not None]

    out_path = os.path.join(base, "files", f"output_{MODEL_ID}.csv")

    # --- LOGICA DI RIPRESA ---
    existing_ids = set()
    output: list[Entry] = []
    if os.path.exists(out_path):
        print(f"Trovato output esistente: {out_path}")
        with open(out_path, newline="") as f:
            for r in csv.DictReader(f):
                p = parse_entry(r)
                if p is not None:
                    existing_ids.add(p.id)
                    output.append(p)
        original = len(entries)
        entries = [e for e in entries if e.id not in existing_ids]
        print(f"Saltate {original - len(entries)} entry già processate.")
        if not entries:
            print("Tutte le entry già processate. Uscita.")
            return

    random.shuffle(entries)
    entries = entries[: HISTORY_PER_REQUEST * NUM_REQUESTS]

    if DRY_RUN:
        print("DRY_RUN=True — nessuna richiesta API.")
        save_output(out_path, output + entries)
        return

    batches = [
        entries[i : i + HISTORY_PER_REQUEST]
        for i in range(0, len(entries), HISTORY_PER_REQUEST)
    ]
    total = len(batches)

    print(f"Avvio {total} batch. Pacing: 1 batch ogni {LAUNCH_INTERVAL_SEC}s.")

    # Usiamo un callback per salvare i risultati non appena un thread finisce
    def on_done(future):
        try:
            results = future.result()
            for entry, v in results:
                output.append(replace(entry, v_llm=v))
            save_output(out_path, output)
            print(f"  + Salvato. Totale entry accumulate: {len(output)}")
        except Exception as e:
            print(f"  Errore nel thread: {e}")

    with ThreadPoolExecutor(max_workers=MAX_INFLIGHT) as pool:
        futures = []
        for batch_idx, batch in enumerate(batches, start=1):
            # Sottometti il task al pool
            fut = pool.submit(
                process_batch, client, base_prompt, base, batch, batch_idx, total
            )
            fut.add_done_callback(on_done)
            futures.append(fut)

            # Se non è l'ultimo, aspetta 80 secondi prima di lanciare il prossimo
            if batch_idx < total:
                time.sleep(LAUNCH_INTERVAL_SEC)

        print(
            "\nTutti i batch sono stati lanciati. Attendo il completamento delle risposte in corso..."
        )
        # Aspetta che tutti i future terminino
        for fut in futures:
            fut.result()

    print(f"\nFatto. {len(output)} entry totali in {out_path}")


if __name__ == "__main__":
    run()
