from dataclasses import dataclass, replace
import random
import csv
import json
import os
import re
import time
import ollama

DRY_RUN = False
HISTORY_PER_REQUEST = 1
NUM_REQUESTS = 300
RETRY_LIMIT = 3
RETRY_DELAY = 5
MODEL_ID = "granite4.1:8b"
TEMPERATURE = 0.7


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


def read_history(base: str, path: str) -> str | None:
    full = os.path.join(base, "files", path)
    if full not in _file_cache:
        try:
            _file_cache[full] = open(full).read()
        except FileNotFoundError:
            print(f"  file non trovato: {full}")
            return None
    return _file_cache[full]


def call_with_retry(prompt, retries=RETRY_LIMIT, delay=RETRY_DELAY) -> str | None:
    for attempt in range(retries):
        try:
            response = ollama.chat(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": TEMPERATURE},
            )
            content = response["message"]["content"]
            print(
                f"  risposta grezza: {content}..."
            )  # Mostra solo l'inizio per non sporcare il terminale

            if content and content.strip():
                return content
            print(f"  risposta vuota (tentativo {attempt + 1}/{retries})")
        except Exception as e:
            print(f"  errore richiesta Ollama (tentativo {attempt + 1}/{retries}): {e}")
        if attempt < retries - 1:
            time.sleep(delay)
    return None


def load_documentation(base: str) -> tuple[str, str]:
    doc = open(os.path.join(base, "docs/MiniGridDocumentation2.md")).read()
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


def save_output(path: str, entries: list[Entry]):
    fieldnames = list(Entry.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            writer.writerow(e.__dict__)


def run():
    print(f"Utilizzo modello locale Ollama: {MODEL_ID}")

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc, vfun_def = load_documentation(base)
    base_prompt = build_base_prompt(base, doc, vfun_def)

    with open(os.path.join(base, "files/metadata.csv")) as f:
        entries = [p for r in csv.DictReader(f) if (p := parse_entry(r)) is not None]

    out_path = os.path.join(base, "files", f"metadata_output_{MODEL_ID}.csv")

    # --- LOGICA DI RIPRESA ---
    existing_ids = set()
    output: list[Entry] = []

    if os.path.exists(out_path):
        print(
            f"Trovato file di output esistente: {out_path}. Caricamento dati salvati..."
        )
        with open(out_path, newline="") as f:
            for r in csv.DictReader(f):
                p = parse_entry(r)
                if p is not None:
                    existing_ids.add(p.id)
                    output.append(p)

        original_count = len(entries)
        entries = [e for e in entries if e.id not in existing_ids]
        print(f"Saltate {original_count - len(entries)} entry già processate.")

        if not entries:
            print("Tutte le entry sono già state processate. Uscita.")
            return
    # -------------------------

    random.shuffle(entries)
    entries = entries[: HISTORY_PER_REQUEST * NUM_REQUESTS]

    if DRY_RUN:
        print("DRY_RUN=True — nessuna richiesta API effettuata.")
        print(
            f"Storico per richiesta: {HISTORY_PER_REQUEST} entry, {NUM_REQUESTS} richieste, {len(entries)} entry totali da processare"
        )
        print(f"Entry selezionate ({len(entries)}):")
        for e in entries:
            print(f"  {e.id}: seed={e.seed}, event={e.event}, v_value={e.v_value}")

        print("\nVerifica file history (primo batch):")
        test_batch = entries[:HISTORY_PER_REQUEST]
        for h in test_batch:
            content = read_history(base, h.path)
            status = "OK" if content is not None else "MANCANTE"
            print(f"  {h.id:6s} → files/{h.path}  [{status}]")

        prompt = build_history_prompt(base_prompt, base, test_batch)
        print(
            f"\nPrompt build: {len(prompt)} caratteri, {HISTORY_PER_REQUEST} history inclusi"
        )

        save_output(out_path, output + entries)
        print(f"\nCSV template salvato in {out_path}")
        return

    batches = [
        entries[i : i + HISTORY_PER_REQUEST]
        for i in range(0, len(entries), HISTORY_PER_REQUEST)
    ]

    for batch_idx, batch in enumerate(batches, start=1):
        prompt = build_history_prompt(base_prompt, base, batch)

        ids = ", ".join(e.id for e in batch)
        print(f"\n[Batch {batch_idx}/{len(batches)}] ({ids})")

        print(f"  richiesta in corso... ({len(prompt)} caratteri)")
        response = call_with_retry(prompt)
        if response is None:
            print(f"  fallito dopo {RETRY_LIMIT} tentativi, salto batch")
            continue

        try:
            results = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\[\s*\{.*\}\s*\]", response, re.DOTALL)
            if match:
                try:
                    results = json.loads(match.group())
                except json.JSONDecodeError:
                    print(f"  errore parsing JSON (anche dopo estrazione)")
                    continue
            else:
                print(f"  errore parsing JSON: nessun array JSON trovato")
                continue

        if not isinstance(results, list):
            print(f"  risposta non è una lista: {type(results).__name__}")
            continue

        for item in results:
            code = item.get("code", "").removeprefix("history_")
            match = next((e for e in batch if e.id == code), None)
            if match is None:
                print(f"  code {code} non trovato nel batch, salto")
                continue
            try:
                v = float(item["v-function-value"])
            except (KeyError, ValueError) as e:
                print(f"  value mancante o invalido per code {code}: {e}")
                continue
            output.append(replace(match, v_llm=v))

        save_output(out_path, output)
        print(f"  batch completato: {len(results)} valori → salvato")

    print(f"\nFatto. {len(output)} entry totali in {out_path}")


if __name__ == "__main__":
    run()
