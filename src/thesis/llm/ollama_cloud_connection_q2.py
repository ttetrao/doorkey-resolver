from dataclasses import dataclass
import random
import csv
import json
import os
import re
import time
from ollama import Client

DRY_RUN = False
HISTORY_PER_REQUEST = 5
NUM_REQUESTS = 30
RETRY_LIMIT = 3
RETRY_DELAY = 2

# Modello su Ollama Cloud
MODEL_ID = "gpt-oss:120b"

# Nome del file di output in base al modello
SAFE_MODEL_NAME = MODEL_ID.replace(":", "_")
OUT_FILENAME = f"output_q_ollama_{SAFE_MODEL_NAME}.csv"

# Azioni da ignorare
IGNORED_ACTIONS = ["drop"]


@dataclass(frozen=True)
class StateEntry:
    """Rappresenta la riga (s, a) con entry_type=1 letta dal CSV."""

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
    q_value: float  # Arriva direttamente dal CSV!


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
            action=raw["action"],
            q_value=float(raw["q_value"]),
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


def call_with_retry(
    client, prompt, retries=RETRY_LIMIT, delay=RETRY_DELAY
) -> str | None:
    for attempt in range(retries):
        try:
            response = client.chat(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": 0.5},
            )

            content = response["message"]["content"]
            print(f"  risposta grezza: {content}...")  # Anteprima per log

            if content and content.strip():
                return content
            print(f"  risposta vuota (tentativo {attempt + 1}/{retries})")
        except Exception as e:
            print(f"  errore richiesta (tentativo {attempt + 1}/{retries}): {e}")
        if attempt < retries - 1:
            time.sleep(delay)
    return None


def load_documentation(base: str) -> tuple[str, str]:
    doc = open(os.path.join(base, "docs/MiniGridDocumentation2.md")).read()
    vfun_def = open(os.path.join(base, "docs/en/v-function_definition.md")).read()
    return doc, vfun_def


def build_base_prompt(base: str, doc: str, vfun_def: str) -> str:
    tmpl = open(os.path.join(base, "docs/en/Prompt-q2-en.txt")).read()
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
        # Nidifichiamo l'azione all'interno della history!
        parts.append(
            f"\n<history_{h.id}>\n{content}\n<action>{h.action}</action>\n</history_{h.id}>"
        )
    return "".join(parts)


def save_output(path: str, entries: list[QEntry]):
    fieldnames = list(QEntry.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            writer.writerow(e.__dict__)


def parse_response(response, batch: list[StateEntry]) -> list[QEntry]:
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

        action = item.get("action", "").lower().strip()
        if not action or action in IGNORED_ACTIONS:
            continue

        try:
            q_llm = float(item.get("v_function_value", 0.0))
        except (ValueError, TypeError):
            continue

        # q_value reale è già dentro match_state, non serve la lookup table!
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
                q_value=match_state.q_value,
                q_llm=q_llm,
            )
        )
    return out


def run():
    client = Client(
        host=os.environ.get("OLLAMA_HOST", "https://ollama.com"),
        headers={"Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY", "")},
    )

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc, vfun_def = load_documentation(base)
    base_prompt = build_base_prompt(base, doc, vfun_def)

    states: list[StateEntry] = []

    # Leggiamo il metadata.csv e prendiamo SOLO le entry_type=1
    with open(os.path.join(base, "files/metadata.csv")) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["entry_type"] == "1":
                state = parse_state_entry(r)
                if state and state.action not in IGNORED_ACTIONS:
                    states.append(state)

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
        print(f"Saltate {original - len(states)} entry (s,a) già processate.")
        if not states:
            print("Tutte le entry già processate. Uscita.")
            return

    random.shuffle(states)
    states = states[: HISTORY_PER_REQUEST * NUM_REQUESTS]

    if DRY_RUN:
        print("DRY_RUN=True — nessuna richiesta API effettuata.")
        print(
            f"Storico per richiesta: {HISTORY_PER_REQUEST} entry, {NUM_REQUESTS} richieste, {len(states)} entry totali da processare"
        )
        print(f"Entry selezionate ({len(states)}):")
        for e in states:
            print(f"  {e.id}: seed={e.seed}, event={e.event}, action={e.action}")

        print("\nVerifica file history (primo batch):")
        test_batch = states[:HISTORY_PER_REQUEST]
        for h in test_batch:
            content = read_history(base, h.path)
            status = "OK" if content is not None else "MANCANTE"
            print(f"  {h.id:6s} → files/{h.path}  [{status}]")

        prompt = build_history_prompt(base_prompt, base, test_batch)
        print(
            f"\nPrompt build: {len(prompt)} caratteri, {HISTORY_PER_REQUEST} history inclusi"
        )
        return

    batches = [
        states[i : i + HISTORY_PER_REQUEST]
        for i in range(0, len(states), HISTORY_PER_REQUEST)
    ]

    for batch_idx, batch in enumerate(batches, start=1):
        prompt = build_history_prompt(base_prompt, base, batch)

        ids = ", ".join(e.id for e in batch)
        print(f"[Batch {batch_idx}/{len(batches)}] ({ids})")

        print(f"  richiesta... ({len(prompt)} caratteri)")
        response = call_with_retry(client, prompt)
        print(f"raw: {response}")
        if response is None:
            print(f"  fallito dopo {RETRY_LIMIT} tentativi, salto batch")
            continue

        parsed_entries = parse_response(response, batch)

        if not parsed_entries:
            print(f"  nessun dato valido estratto per il batch {batch_idx}")
            continue

        for entry in parsed_entries:
            output.append(entry)

        save_output(out_path, output)
        print(f"  batch completato: {len(parsed_entries)} valori → salvato")

    print(f"\nFatto. {len(output)} entry totali in {out_path}")


if __name__ == "__main__":
    run()
