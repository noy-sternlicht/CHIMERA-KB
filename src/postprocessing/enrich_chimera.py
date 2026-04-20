"""
enrich_chimera.py
-----------------
1. Downloads the CHIMERA dataset from HuggingFace (noystl/CHIMERA).
2. Renders the postprocessing Jinja2 prompt for each row.
3. Calls an LLM (Claude or GPT) to produce enriched entity descriptions.
4. Saves the enriched dataset as a new CSV next to this script.

API keys are read from a secrets.yaml file (default: <repo-root>/secrets.yaml):
    anthropic_api_key: "sk-ant-..."
    openai_api_key:    "sk-..."

Usage:
    python enrich_chimera.py \
        --model    claude-3-5-sonnet-20241022      # or any gpt-* / o* model name
        --output   chimera_enriched.csv  \
        --max_rows 0                              # 0 = process all rows
        --secrets  /path/to/secrets.yaml          # optional override

Progress is checkpointed: if --output already exists, already-processed rows
are skipped so the script can be safely restarted.
"""

import argparse
import json
import logging
import platform
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import anthropic
import openai
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from jinja2 import Environment, FileSystemLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing  (USD per million tokens – update as needed)
# https://www.anthropic.com/pricing  /  https://openai.com/api/pricing
# ---------------------------------------------------------------------------
ANTHROPIC_PRICING = {
    "default":          {"input": 3.00,  "output": 15.00, "cache_write": 3.75,  "cache_read": 0.30},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10},
    "claude-sonnet-4-6": {"input": 3.00,  "output":  15.00, "cache_write": 3.75 ,  "cache_read": 0.30},
    "claude-opus-4-6":    {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
}

OPENAI_PRICING = {
    "gpt-4.1":       {"input": 2.00,  "output":  8.00, "cached": 0.50},
    "gpt-4.1-mini":  {"input": 0.40,  "output":  1.60, "cached": 0.10},
    "gpt-4.1-nano":  {"input": 0.10,  "output":  0.40, "cached": 0.025},
    "gpt-4o-mini":   {"input": 0.15,  "output":  0.60, "cached": 0.075},
    "gpt-4o":        {"input": 2.50,  "output": 10.00, "cached": 1.25},
    "o3-mini":       {"input": 1.10,  "output":  4.40, "cached": 0.275},
    "o3":            {"input": 10.00, "output": 40.00, "cached": 2.50},
    "o4-mini":       {"input": 1.10,  "output":  4.40, "cached": 0.275},
    "gpt-5":         {"input": 1.25,  "output": 10.00, "cached": 0.125},
    "gpt-5.1":       {"input": 1.25,  "output": 10.00, "cached": 0.125},
    "gpt-5.2":       {"input": 1.75,  "output": 14.00, "cached": 0.155},
    "gpt-5.4":       {"input": 2.50,  "output": 15.00, "cached": 0.25},
    "gpt-5.4-mini":  {"input": 0.75,  "output":  4.50, "cached": 0.075},
    "gpt-5.4-nano":  {"input": 0.20,  "output":  1.25, "cached": 0.02},
    "default":       {"input": 2.50,  "output": 10.00, "cached": 1.25},
}


def _get_anthropic_pricing(model: str) -> dict:
    model_lower = model.lower()
    for key in ANTHROPIC_PRICING:
        if key != "default" and key in model_lower:
            return ANTHROPIC_PRICING[key]
    return ANTHROPIC_PRICING["default"]


def _get_openai_pricing(model: str) -> dict:
    model_lower = model.lower()
    best_key, best_len = "default", 0
    for key in OPENAI_PRICING:
        if key != "default" and key in model_lower and len(key) > best_len:
            best_key, best_len = key, len(key)
    return OPENAI_PRICING[best_key]


class CostTracker:
    """Thread-safe token-usage accumulator for Anthropic and OpenAI."""

    def __init__(self):
        self._lock = threading.Lock()
        self._anthropic = {"input_tokens": 0, "output_tokens": 0,
                           "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        self._openai = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

    def add_anthropic_usage(self, usage):
        def _get(obj, key):
            return (obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)) or 0
        with self._lock:
            self._anthropic["input_tokens"] += _get(usage, "input_tokens")
            self._anthropic["output_tokens"] += _get(usage, "output_tokens")
            self._anthropic["cache_creation_input_tokens"] += _get(usage, "cache_creation_input_tokens")
            self._anthropic["cache_read_input_tokens"] += _get(usage, "cache_read_input_tokens")

    def add_openai_usage(self, usage):
        def _get(obj, key):
            return (obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)) or 0
        def _cached(obj):
            details = (obj.get("prompt_tokens_details") if isinstance(obj, dict)
                       else getattr(obj, "prompt_tokens_details", None))
            if details is None:
                return 0
            return (details.get("cached_tokens") if isinstance(details, dict)
                    else getattr(details, "cached_tokens", None)) or 0
        with self._lock:
            self._openai["input_tokens"] += _get(usage, "prompt_tokens")
            self._openai["output_tokens"] += _get(usage, "completion_tokens")
            self._openai["cached_tokens"] += _cached(usage)

    def compute_anthropic_cost(self, model: str) -> dict:
        p = _get_anthropic_pricing(model)
        u = dict(self._anthropic)
        costs = {
            "input":       u["input_tokens"]                    * p["input"]       / 1_000_000,
            "output":      u["output_tokens"]                   * p["output"]      / 1_000_000,
            "cache_write": u["cache_creation_input_tokens"]     * p["cache_write"] / 1_000_000,
            "cache_read":  u["cache_read_input_tokens"]         * p["cache_read"]  / 1_000_000,
        }
        return {"usage": u, "costs": costs, "total": sum(costs.values()), "pricing": p}

    def compute_openai_cost(self, model: str) -> dict:
        p = _get_openai_pricing(model)
        u = dict(self._openai)
        non_cached = max(0, u["input_tokens"] - u["cached_tokens"])
        costs = {
            "non_cached_input": non_cached           * p["input"]  / 1_000_000,
            "cached_input":     u["cached_tokens"]   * p["cached"] / 1_000_000,
            "output":           u["output_tokens"]   * p["output"] / 1_000_000,
        }
        return {"usage": u, "costs": costs, "total": sum(costs.values()), "pricing": p,
                "non_cached_input_tokens": non_cached}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _find_repo_root() -> Path:
    """Walk up from this file until we find the .git directory."""
    current = Path(__file__).resolve().parent
    for directory in [current, *current.parents]:
        if (directory / ".git").exists():
            return directory
    raise RuntimeError(f"Could not locate repository root (no .git found above {current})")


REPO_ROOT = _find_repo_root()
PROMPT_FILE = Path(__file__).parent / "postprocessing_prompt.jinja2"
HF_REPO = "noystl/CHIMERA"
HF_FILENAME = "raw_edges.csv"


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

def load_secrets(secrets_path: Path) -> dict:
    if not secrets_path.exists():
        raise FileNotFoundError(
            f"secrets.yaml not found at {secrets_path}. "
            "Copy secrets.yaml.example to secrets.yaml and fill in your keys."
        )
    with open(secrets_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompt_template():
    env = Environment(loader=FileSystemLoader(str(PROMPT_FILE.parent)))
    env.filters["tojson"] = lambda v, indent=None: json.dumps(v, indent=indent, ensure_ascii=False)
    return env.get_template(PROMPT_FILE.name)




def row_to_recombination(row: pd.Series) -> dict:
    """Convert a flat CSV row into the nested recombination dict expected by the prompt."""
    relation = row["relation"]
    if relation == "combination":
        return {
            "type": "combination",
            "entities": {
                "comb-element": [row["source_text"], row["target_text"]],
            },
        }
    elif relation == "inspiration":
        return {
            "type": "inspiration",
            "entities": {
                "inspiration-src": [row["target_text"]],
                "inspiration-target": [row["source_text"]],
            },
        }
    else:
        raise ValueError(f"Unknown relation type: {relation!r}")


def _is_claude_model(model: str) -> bool:
    return model.startswith("claude")


def call_llm(
    model: str,
    prompt: str,
    claude_client: Optional[anthropic.Anthropic],
    openai_client: Optional[openai.OpenAI],
    cost_tracker: CostTracker,
    max_tokens: int = 1024,
    retries: int = 4,
    backoff: float = 5.0,
) -> str:
    """Dispatch to the appropriate provider and retry on transient errors."""
    if _is_claude_model(model):
        return _call_claude(claude_client, prompt, model, cost_tracker, max_tokens, retries, backoff)
    else:
        return _call_openai(openai_client, prompt, model, cost_tracker, max_tokens, retries, backoff)


def _call_claude(
    client: anthropic.Anthropic,
    prompt: str,
    model: str,
    cost_tracker: CostTracker,
    max_tokens: int,
    retries: int,
    backoff: float,
) -> str:
    for attempt in range(1, retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            cost_tracker.add_anthropic_usage(response.usage)
            if not response.content:
                raise ValueError(f"Empty content list from Claude (stop_reason={response.stop_reason!r})")
            return response.content[0].text
        except anthropic.RateLimitError as exc:
            wait = backoff * attempt
            logger.warning(f"Claude rate-limited (attempt {attempt}/{retries}). Waiting {wait}s… ({exc})")
            time.sleep(wait)
        except anthropic.APIStatusError as exc:
            logger.warning(f"Claude API error on attempt {attempt}/{retries}: {exc}")
            if attempt == retries:
                raise
            time.sleep(backoff)
    raise RuntimeError("All Claude retry attempts exhausted.")


def _call_openai(
    client: openai.OpenAI,
    prompt: str,
    model: str,
    cost_tracker: CostTracker,
    max_tokens: int,
    retries: int,
    backoff: float,
) -> str:
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            cost_tracker.add_openai_usage(response.usage)
            return response.choices[0].message.content
        except openai.RateLimitError as exc:
            wait = backoff * attempt
            logger.warning(f"OpenAI rate-limited (attempt {attempt}/{retries}). Waiting {wait}s… ({exc})")
            time.sleep(wait)
        except openai.APIStatusError as exc:
            logger.warning(f"OpenAI API error on attempt {attempt}/{retries}: {exc}")
            if attempt == retries:
                raise
            time.sleep(backoff)
    raise RuntimeError("All OpenAI retry attempts exhausted.")


def parse_enriched(raw: str, relation: str) -> dict:
    """
    Extract the JSON object from the model response and return
    {'source_text_processed': ..., 'target_text_processed': ...}.
    Falls back to None values on parse failure.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group())
            except json.JSONDecodeError:
                logger.warning("Could not parse JSON from model output.")
                return {"source_text_processed": None, "target_text_processed": None}
        else:
            logger.warning("No JSON found in model output.")
            return {"source_text_processed": None, "target_text_processed": None}

    entities = obj.get("entities", {})
    if relation == "combination":
        elements = entities.get("comb-element", [None, None])
        return {
            "source_text_processed": elements[0] if len(elements) > 0 else None,
            "target_text_processed": elements[1] if len(elements) > 1 else None,
        }
    else:  # inspiration
        src_list = entities.get("inspiration-src", [None])
        tgt_list = entities.get("inspiration-target", [None])
        # inspiration-src = target_text, inspiration-target = source_text
        return {
            "source_text_processed": tgt_list[0] if tgt_list else None,
            "target_text_processed": src_list[0] if src_list else None,
        }


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce column ordering: relation after id; source/target processed before originals."""
    cols = list(df.columns)

    # Move 'relation' to right after 'id'
    if "relation" in cols and "id" in cols:
        cols.remove("relation")
        cols.insert(cols.index("id") + 1, "relation")

    # Group source/target columns as: processed first, then originals
    text_group = [c for c in ["source_text_processed", "target_text_processed", "source_text", "target_text"] if c in cols]
    if text_group:
        insert_at = min(cols.index(c) for c in text_group if c in cols)
        for c in text_group:
            cols.remove(c)
        for i, c in enumerate(text_group):
            cols.insert(insert_at + i, c)

    return df[cols]


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def _build_batch_requests(
    pending: pd.DataFrame,
    template,
) -> List[Tuple[str, str, str, str]]:
    """
    Render prompts for all pending rows.
    Returns list of (row_id, custom_id, prompt, relation) tuples.
    Rows with unknown relation types are skipped with a warning.
    """
    requests = []
    for _, row in pending.iterrows():
        try:
            recombination = row_to_recombination(row)
        except ValueError as e:
            logger.warning(f"Skipping row {row['id']}: {e}")
            continue
        prompt = template.render(
            abstract=row["abstract"],
            recombination=recombination,
        )
        row_id = str(row["id"])
        requests.append((row_id, f"row_{row_id}", prompt, row["relation"]))
    return requests


def _submit_anthropic_chunks(
    client: anthropic.Anthropic,
    model: str,
    requests: List[Tuple[str, str, str, str]],
    chunk_size: int,
) -> List[dict]:
    chunks = []
    for i in range(0, len(requests), chunk_size):
        chunk = requests[i : i + chunk_size]
        chunk_index = len(chunks)
        batch_requests = [
            {
                "custom_id": custom_id,
                "params": {
                    "model": model,
                    "max_tokens": 1024,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
            for (_, custom_id, prompt, _) in chunk
        ]
        batch = client.messages.batches.create(requests=batch_requests)
        row_ids = [row_id for (row_id, _, _, _) in chunk]
        chunks.append({
            "chunk_index": chunk_index,
            "batch_id": batch.id,
            "status": "in_progress",
            "retrieved": False,
            "row_ids": row_ids,
        })
        logger.info(f"  Submitted chunk {chunk_index}: {len(chunk):,} rows → batch_id={batch.id}")
    return chunks


def _submit_openai_chunks(
    client: openai.OpenAI,
    model: str,
    requests: List[Tuple[str, str, str, str]],
    chunk_size: int,
    output_path: Path,
) -> List[dict]:
    from util import request_openai_batch_completions  # src/ is on PYTHONPATH
    chunks = []
    for i in range(0, len(requests), chunk_size):
        chunk = requests[i : i + chunk_size]
        chunk_index = len(chunks)
        prompts_dict = {custom_id: prompt for (_, custom_id, prompt, _) in chunk}
        batch_id = request_openai_batch_completions(
            prompts=prompts_dict,
            max_tokens=1024,
            temperature=0.0,
            batch_idx=chunk_index,
            output_path=str(output_path.parent),
            client=client,
            engine=model,
        )
        row_ids = [row_id for (row_id, _, _, _) in chunk]
        chunks.append({
            "chunk_index": chunk_index,
            "batch_id": batch_id,
            "status": "in_progress",
            "retrieved": False,
            "row_ids": row_ids,
        })
        logger.info(f"  Submitted chunk {chunk_index}: {len(chunk):,} rows → batch_id={batch_id}")
    return chunks


def _check_anthropic_chunks(client: anthropic.Anthropic, chunks: List[dict]) -> List[dict]:
    for chunk in chunks:
        if chunk["retrieved"]:
            continue
        batch = client.messages.batches.retrieve(chunk["batch_id"])
        chunk["status"] = batch.processing_status  # "in_progress" | "ended"
    return chunks


def _check_openai_chunks(client: openai.OpenAI, chunks: List[dict]) -> List[dict]:
    for chunk in chunks:
        if chunk["retrieved"]:
            continue
        batch = client.batches.retrieve(chunk["batch_id"])
        chunk["status"] = batch.status
    return chunks


def _retrieve_anthropic_results(
    client: anthropic.Anthropic,
    chunks: List[dict],
    df: pd.DataFrame,
    cost_tracker: CostTracker,
) -> List[dict]:
    id_to_row = {str(row["id"]): row for _, row in df.iterrows()}
    results = []
    for chunk in chunks:
        if chunk["status"] != "ended" or chunk["retrieved"]:
            continue
        logger.info(f"  Retrieving chunk {chunk['chunk_index']} (batch {chunk['batch_id']}) …")
        for result in client.messages.batches.results(chunk["batch_id"]):
            row_id = result.custom_id.removeprefix("row_")
            row = id_to_row.get(row_id)
            if row is None:
                logger.warning(f"Unknown row_id in batch result: {row_id}")
                continue
            if result.result.type == "succeeded" and result.result.message.content:
                text = result.result.message.content[0].text
                cost_tracker.add_anthropic_usage(result.result.message.usage)
                enriched = parse_enriched(text, row["relation"])
            else:
                if result.result.type == "succeeded":
                    logger.warning(f"Batch result {result.custom_id}: succeeded but empty content (stop_reason={result.result.message.stop_reason})")
                elif result.result.type == "errored":
                    err = result.result.error.error
                    logger.warning(f"Batch result {result.custom_id}: errored — {err.type}: {err.message}")
                else:
                    logger.warning(f"Batch result {result.custom_id}: {result.result.type}")
                enriched = {"source_text_processed": None, "target_text_processed": None}
            result_row = row.to_dict()
            result_row.update(enriched)
            results.append(result_row)
        chunk["retrieved"] = True
    return results


def _retrieve_openai_results(
    client: openai.OpenAI,
    chunks: List[dict],
    df: pd.DataFrame,
) -> List[dict]:
    from util import get_openai_batch_completions  # src/ is on PYTHONPATH
    id_to_row = {str(row["id"]): row for _, row in df.iterrows()}
    results = []
    for chunk in chunks:
        if chunk["status"] != "completed" or chunk["retrieved"]:
            continue
        logger.info(f"  Retrieving chunk {chunk['chunk_index']} (batch {chunk['batch_id']}) …")
        try:
            responses, _ = get_openai_batch_completions(chunk["batch_id"], client)
        except Exception as e:
            logger.error(f"Failed to retrieve chunk {chunk['chunk_index']}: {e}")
            continue
        for custom_id, text in responses.items():
            row_id = custom_id.removeprefix("row_")
            row = id_to_row.get(row_id)
            if row is None:
                logger.warning(f"Unknown row_id in batch result: {row_id}")
                continue
            enriched = parse_enriched(text, row["relation"])
            result_row = row.to_dict()
            result_row.update(enriched)
            results.append(result_row)
        chunk["retrieved"] = True
    return results


def run_batch_mode(
    args: argparse.Namespace,
    df: pd.DataFrame,
    pending: pd.DataFrame,
    template,
    claude_client: Optional[anthropic.Anthropic],
    openai_client: Optional[openai.OpenAI],
    output_path: Path,
    done_df: pd.DataFrame,
    cost_tracker: CostTracker,
) -> Tuple[List[dict], Optional[str], Optional[str]]:
    """
    Batch mode entry point.
    - No state file → submit phase: build prompts, chunk, submit batches, save state.
    - State file exists → check/retrieve phase: poll status, retrieve completed chunks,
      merge into output CSV, update state.
    """
    state_path = output_path.with_name(output_path.stem + "_batch_state.json")
    is_claude = _is_claude_model(args.model)

    if not state_path.exists():
        # ---- SUBMIT PHASE ----
        if len(pending) == 0:
            logger.info("No pending rows to submit.")
            return [], None, None

        logger.info(f"Building prompts for {len(pending):,} pending rows …")
        requests = _build_batch_requests(pending, template)
        logger.info(f"Submitting {len(requests):,} requests in chunks of {args.batch_chunk_size} …")

        if is_claude:
            chunks = _submit_anthropic_chunks(claude_client, args.model, requests, args.batch_chunk_size)
        else:
            chunks = _submit_openai_chunks(openai_client, args.model, requests, args.batch_chunk_size, output_path)

        state = {
            "model": args.model,
            "provider": "anthropic" if is_claude else "openai",
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "chunks": chunks,
        }
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        total_rows = sum(len(c["row_ids"]) for c in chunks)
        logger.info(
            f"Submitted {len(chunks)} chunk(s) covering {total_rows:,} rows. "
            f"State saved → {state_path}\n"
            "Run again with the same arguments to check status and retrieve results."
        )
        return [], None, None

    else:
        # ---- CHECK / RETRIEVE PHASE ----
        try:
            with open(state_path) as f:
                state = json.load(f)
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"State file is empty or corrupt ({state_path}). Deleting and re-submitting.")
            state_path.unlink()
            return run_batch_mode(args, df, pending, template, claude_client, openai_client, output_path, done_df, cost_tracker)

        chunks = state["chunks"]
        logger.info(f"Checking status of {len(chunks)} chunk(s) …")

        if is_claude:
            chunks = _check_anthropic_chunks(claude_client, chunks)
        else:
            chunks = _check_openai_chunks(openai_client, chunks)

        # Submit any pending rows not yet covered by existing chunks
        if getattr(args, "retrieve_only", False):
            logger.info("--retrieve_only: skipping submission of uncovered rows.")
        else:
            covered_ids = {row_id for c in chunks for row_id in c["row_ids"]}
            uncovered = pending[~pending["id"].astype(str).isin(covered_ids)]
            if not uncovered.empty:
                logger.info(f"Submitting {len(uncovered):,} additional rows not covered by existing chunks …")
                new_requests = _build_batch_requests(uncovered, template)
                if is_claude:
                    new_chunks = _submit_anthropic_chunks(claude_client, args.model, new_requests, args.batch_chunk_size)
                else:
                    new_chunks = _submit_openai_chunks(openai_client, args.model, new_requests, args.batch_chunk_size, output_path)
                # Re-index chunk indices to continue from where existing ones left off
                offset = max(c["chunk_index"] for c in chunks) + 1
                for c in new_chunks:
                    c["chunk_index"] += offset
                chunks.extend(new_chunks)

        # Retrieve any newly completed chunks
        if is_claude:
            new_results = _retrieve_anthropic_results(claude_client, chunks, df, cost_tracker)
        else:
            new_results = _retrieve_openai_results(openai_client, chunks, df)

        debug_entries = []
        example_prompt: Optional[str] = None
        example_row_id: Optional[str] = None

        if new_results:
            batch_df = pd.DataFrame(new_results)
            combined = pd.concat([done_df, batch_df], ignore_index=True) if not done_df.empty else batch_df
            _reorder_columns(combined).to_csv(output_path, index=False)
            logger.info(f"Saved {len(new_results):,} new rows → {output_path} ({len(combined):,} total)")

            for result_row in new_results:
                row = pd.Series(result_row)
                try:
                    recombination = row_to_recombination(row)
                except ValueError:
                    continue
                enriched = {
                    "source_text_processed": result_row.get("source_text_processed"),
                    "target_text_processed": result_row.get("target_text_processed"),
                }
                if example_prompt is None:
                    example_prompt = template.render(
                        abstract=row["abstract"],
                        recombination=recombination,
                    )
                    example_row_id = str(result_row.get("id"))
                debug_entries.append({
                    "id": result_row.get("id"),
                    "paper_id": result_row.get("paper_id"),
                    "relation": result_row.get("relation"),
                    "abstract": result_row.get("abstract"),
                    "before": recombination["entities"],
                    "after": enriched,
                })
        else:
            logger.info("No new results to save this run.")

        # Status summary table
        all_retrieved = all(c["retrieved"] for c in chunks)
        submitted_at = state.get("submitted_at", "unknown")
        header = (
            f"\n  {'#':>3}  {'Batch ID':<38}  {'Status':<15}  {'Retrieved':<10}  {'Rows':>5}"
        )
        sep = f"  {'-'*3}  {'-'*38}  {'-'*15}  {'-'*10}  {'-'*5}"
        rows_lines = []
        for c in chunks:
            status_str = c["status"]
            retrieved_str = "yes" if c["retrieved"] else ("ready" if c["status"] == "ended" else "waiting")
            rows_lines.append(
                f"  {c['chunk_index']:>3}  {c['batch_id']:<38}  {status_str:<15}  {retrieved_str:<10}  {len(c['row_ids']):>5}"
            )
        newly_retrieved = len(new_results)
        total_rows = sum(len(c["row_ids"]) for c in chunks)
        done_chunks = sum(1 for c in chunks if c["retrieved"])
        logger.info(
            f"Batch status (submitted {submitted_at}):"
            f"{header}\n{sep}\n" + "\n".join(rows_lines) +
            f"\n{sep}"
            f"\n  Chunks: {done_chunks}/{len(chunks)} retrieved  |  "
            f"Rows: {newly_retrieved} new this run / {total_rows} total in batch"
        )

        # Persist updated state (status + retrieved flags), or clean up if fully done
        if all_retrieved:
            state_path.unlink()
            logger.info("All chunks retrieved — state file removed. Next run will submit new batches.")
        else:
            state["chunks"] = chunks
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            pending_count = sum(1 for c in chunks if not c["retrieved"])
            logger.info(f"{pending_count} chunk(s) still in progress — run again to check.")
        return debug_entries, example_prompt, example_row_id


# ---------------------------------------------------------------------------
# Debug report
# ---------------------------------------------------------------------------

DEBUG_CHUNK_SIZE = 50  # max entries per chunk file


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def _debug_entries_jsonl_path(output_path: Path) -> Path:
    """Sidecar JSONL that accumulates all debug entries across runs."""
    return output_path.with_name(output_path.stem + "_debug_entries.jsonl")


def _load_accumulated_entries(output_path: Path) -> list:
    path = _debug_entries_jsonl_path(output_path)
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _append_to_entries_log(new_entries: list, output_path: Path) -> None:
    path = _debug_entries_jsonl_path(output_path)
    with open(path, "a", encoding="utf-8") as f:
        for entry in new_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _chunk_debug_path(output_path: Path, chunk_idx: int, n_chunks: int) -> Path:
    if n_chunks == 1:
        return output_path.with_name(output_path.stem + "_debug.md")
    return output_path.with_name(output_path.stem + f"_debug_{chunk_idx:03d}.md")


def _pre(text: str) -> str:
    """Wrap text in a pre block that soft-wraps long lines."""
    escaped = text.strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<pre style="white-space: pre-wrap; word-wrap: break-word;">{escaped}</pre>'


def _abstract_with_highlights(abstract: str, enriched_values: list) -> str:
    """Return the abstract as a wrapping paragraph with enriched entities bolded."""
    escaped = abstract.strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for val in enriched_values:
        if not val:
            continue
        escaped_val = val.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        escaped = escaped.replace(escaped_val, f"<strong>{escaped_val}</strong>")
    return (
        '<blockquote style="white-space: pre-wrap; word-wrap: break-word; '
        f'border-left: 4px solid #aaa; padding: 0.5em 1em;">{escaped}</blockquote>'
    )


def _build_header_section(
    args: argparse.Namespace,
    prompt_template_text: str,
    example_prompt: Optional[str],
    example_row_id: Optional[str],
    chunk_idx: int,
    n_chunks: int,
    total_entries: int,
) -> list:
    lines = [
        "# Enrichment Debug Report",
        "",
        "## Run Info",
        "",
        f"| Key | Value |",
        f"|---|---|",
        f"| Timestamp | `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}` |",
        f"| Command | `{' '.join(sys.argv)}` |",
        f"| Model | `{args.model}` |",
        f"| Output | `{args.output}` |",
        f"| Max rows | `{args.max_rows}` (`0` = all) |",
        f"| Secrets file | `{args.secrets}` |",
        f"| Dataset | `{HF_REPO} / {HF_FILENAME}` |",
        f"| Python | `{platform.python_version()}` |",
        f"| anthropic | `{_pkg_version('anthropic')}` |",
        f"| openai | `{_pkg_version('openai')}` |",
        f"| pandas | `{_pkg_version('pandas')}` |",
    ]
    if n_chunks > 1:
        lines += [
            f"| Chunk | `{chunk_idx}/{n_chunks}` |",
            f"| Total entries | `{total_entries}` |",
        ]
    lines += [""]

    lines += [
        "## Prompt",
        "",
        "<details>",
        "<summary>Prompt template</summary>",
        "",
        _pre(prompt_template_text),
        "",
        "</details>",
        "",
    ]

    if example_prompt is not None:
        lines += [
            "<details>",
            f"<summary>Rendered prompt — row ID <code>{example_row_id}</code></summary>",
            "",
            _pre(example_prompt),
            "",
            "</details>",
            "",
        ]
    return lines


def _build_entries_section(chunk_entries: list) -> list:
    lines = ["## Results", ""]
    for entry in chunk_entries:
        row_id = entry["id"]
        paper_id = entry["paper_id"]
        relation = entry["relation"]
        abstract = entry["abstract"]
        before = entry["before"]
        after = entry["after"]

        original_values = [v for vals in before.values() for v in (vals if isinstance(vals, list) else [vals])]

        lines += [
            f"### Row ID: `{row_id}` | Paper ID: `{paper_id}` | Relation: `{relation}`",
            "",
            "**Abstract**",
            "",
            _abstract_with_highlights(abstract, original_values),
            "",
            "**Entities before enrichment**",
            "",
        ]
        for k, v in before.items():
            for item in (v if isinstance(v, list) else [v]):
                lines += [f"- `{k}`: {item}", ""]

        lines += [
            "**Entities after enrichment**",
            "",
            f"- `source_text_processed`: {after.get('source_text_processed')}",
            "",
            f"- `target_text_processed`: {after.get('target_text_processed')}",
            "",
            "---",
            "",
        ]
    return lines


def _build_cost_section(
    args: argparse.Namespace,
    cost_tracker: CostTracker,
    total_dataset_rows: int,
    processed_rows: int,
) -> list:
    lines = ["## Cost", ""]
    if _is_claude_model(args.model):
        c = cost_tracker.compute_anthropic_cost(args.model)
        u, costs, p = c["usage"], c["costs"], c["pricing"]
        lines += [
            f"| Token type | Tokens | Rate ($/M) | Cost (USD) |",
            f"|---|---|---|---|",
            f"| Input | {u['input_tokens']:,} | ${p['input']:.2f} | ${costs['input']:.6f} |",
            f"| Output | {u['output_tokens']:,} | ${p['output']:.2f} | ${costs['output']:.6f} |",
            f"| Cache write | {u['cache_creation_input_tokens']:,} | ${p['cache_write']:.2f} | ${costs['cache_write']:.6f} |",
            f"| Cache read | {u['cache_read_input_tokens']:,} | ${p['cache_read']:.2f} | ${costs['cache_read']:.6f} |",
            f"",
            f"**Total cost (this run): ${c['total']:.6f}**",
            f"",
            f"> Rates from https://www.anthropic.com/pricing",
        ]
        actual_total = c["total"]
    else:
        c = cost_tracker.compute_openai_cost(args.model)
        u, costs, p = c["usage"], c["costs"], c["pricing"]
        lines += [
            f"| Token type | Tokens | Rate ($/M) | Cost (USD) |",
            f"|---|---|---|---|",
            f"| Input (non-cached) | {c['non_cached_input_tokens']:,} | ${p['input']:.2f} | ${costs['non_cached_input']:.6f} |",
            f"| Input (cached) | {u['cached_tokens']:,} | ${p['cached']:.3f} | ${costs['cached_input']:.6f} |",
            f"| Output | {u['output_tokens']:,} | ${p['output']:.2f} | ${costs['output']:.6f} |",
            f"",
            f"**Total cost (this run): ${c['total']:.6f}**",
            f"",
            f"> Rates from https://openai.com/api/pricing",
        ]
        actual_total = c["total"]

    if processed_rows > 0 and total_dataset_rows > 0:
        cost_per_row = actual_total / processed_rows
        estimated_total = cost_per_row * total_dataset_rows
        batch_discount = 0.50
        estimated_total_batch = estimated_total * (1 - batch_discount)
        lines += [
            "",
            "## Full-Dataset Cost Estimate",
            "",
            f"| Metric | Standard API | Batch API (−50 %) |",
            f"|---|---|---|",
            f"| Rows processed (this run) | {processed_rows:,} | {processed_rows:,} |",
            f"| Total rows in dataset | {total_dataset_rows:,} | {total_dataset_rows:,} |",
            f"| Cost per row | ${cost_per_row:.6f} | ${cost_per_row * (1 - batch_discount):.6f} |",
            f"| **Estimated total cost** | **${estimated_total:.4f}** | **${estimated_total_batch:.4f}** |",
            f"",
            f"> Extrapolated linearly from {processed_rows:,} processed rows at ${cost_per_row:.6f}/row."
            f" Batch estimate applies the 50 % discount offered by the Anthropic/OpenAI Batch APIs.",
        ]
    return lines


def write_debug_report(
    args: argparse.Namespace,
    prompt_template_text: str,
    example_prompt: Optional[str],
    example_row_id: Optional[str],
    new_debug_entries: list,
    cost_tracker: CostTracker,
    total_dataset_rows: int,
    processed_rows: int,
    output_path: Path,
) -> None:
    """
    Append new_debug_entries to the sidecar JSONL, then (re-)write all chunk MD files.

    - Entries accumulate across runs (never overwritten).
    - Files are split at DEBUG_CHUNK_SIZE entries per chunk.
    - Cost/estimate section is always written to the last chunk only.
    - First chunk always contains the Run Info and Prompt sections.
    """
    # 1. Persist new entries (append-only)
    if new_debug_entries:
        _append_to_entries_log(new_debug_entries, output_path)

    # 2. Load all accumulated entries
    all_entries = _load_accumulated_entries(output_path)
    if not all_entries:
        return

    # 3. Split into chunks of DEBUG_CHUNK_SIZE
    chunks = [all_entries[i:i + DEBUG_CHUNK_SIZE] for i in range(0, len(all_entries), DEBUG_CHUNK_SIZE)]
    n_chunks = len(chunks)
    total_entries = len(all_entries)

    # 4. Write each chunk file
    written_paths = []
    for ci, chunk_entries in enumerate(chunks):
        chunk_num = ci + 1
        is_first = ci == 0
        is_last = ci == n_chunks - 1
        chunk_path = _chunk_debug_path(output_path, chunk_num, n_chunks)

        lines = []
        if is_first:
            lines += _build_header_section(
                args, prompt_template_text, example_prompt, example_row_id,
                chunk_num, n_chunks, total_entries,
            )
        else:
            lines += [
                "# Enrichment Debug Report",
                "",
                f"_(Chunk {chunk_num}/{n_chunks} — entries {ci * DEBUG_CHUNK_SIZE + 1}–{ci * DEBUG_CHUNK_SIZE + len(chunk_entries)} of {total_entries})_",
                "",
            ]

        lines += _build_entries_section(chunk_entries)

        if is_last:
            lines += _build_cost_section(args, cost_tracker, total_dataset_rows, processed_rows)

        chunk_path.write_text("\n".join(lines), encoding="utf-8")
        written_paths.append(chunk_path)

    for path in written_paths:
        logger.info(f"Debug chunk written → {path}")


# ---------------------------------------------------------------------------
# Legacy single-call wrapper kept for call-site compatibility
# ---------------------------------------------------------------------------

def build_debug_md(
    args: argparse.Namespace,
    prompt_template_text: str,
    example_prompt: Optional[str],
    example_row_id: Optional[str],
    debug_entries: list,
    cost_tracker: CostTracker,
    total_dataset_rows: int = 0,
    processed_rows: int = 0,
) -> str:
    """Deprecated: use write_debug_report() instead. Kept for compatibility."""
    lines = _build_header_section(
        args, prompt_template_text, example_prompt, example_row_id,
        chunk_idx=1, n_chunks=1, total_entries=len(debug_entries),
    )
    lines += _build_entries_section(debug_entries)
    lines += _build_cost_section(args, cost_tracker, total_dataset_rows, processed_rows)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _log_enrichment_stats(output_path: Path, total_in_dataset: int) -> None:
    if not output_path.exists():
        logger.info("Enrichment stats: output file does not exist yet.")
        return
    out = pd.read_csv(output_path)
    total = len(out)
    by_relation = out["relation"].value_counts()
    failed_mask = out["source_text_processed"].isna() & out["target_text_processed"].isna()
    n_failed = int(failed_mask.sum())
    n_ok = total - n_failed
    missing = total_in_dataset - total
    lines = [
        f"  Total rows in output : {total:>7,}  /  {total_in_dataset:,} in dataset  ({missing:,} not yet processed)",
        f"  Successfully enriched: {n_ok:>7,}",
        f"  Failed (both null)   : {n_failed:>7,}",
        "  By relation type:",
    ]
    for rel, cnt in by_relation.items():
        lines.append(f"    {rel:<15} {cnt:>7,}")
    logger.info("Enrichment stats:\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Enrich CHIMERA dataset entities via an LLM.")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Model name. Claude models start with 'claude'; GPT models with 'gpt' or 'o'.")
    parser.add_argument("--output", default=str(REPO_ROOT / "output" / "postprocessed"/ "chimera_enriched.csv"), help="Output CSV path.")
    parser.add_argument("--max_rows", type=int, default=0, help="Max rows to process (0 = all).")
    parser.add_argument("--start_from", type=int, default=0, help="Skip the first N rows of the dataset before processing.")
    parser.add_argument("--secrets", default=str(REPO_ROOT / "secrets.yaml"),
                        help="Path to secrets.yaml containing API keys.")
    parser.add_argument("--batch", action="store_true",
                        help="Use the provider's async Batch API (50%% cheaper). "
                             "First run submits; subsequent runs check status and retrieve results.")
    parser.add_argument("--batch_chunk_size", type=int, default=5000,
                        help="Max rows per batch chunk (Anthropic max=10000, OpenAI max=50000). Default: 5000.")
    parser.add_argument("--retrieve_only", action="store_true",
                        help="When --batch is set, only retrieve results from existing batches — never submit new ones.")
    parser.add_argument("--resubmit_failed", action="store_true",
                        help="Re-process rows where both source_text_processed and target_text_processed are null.")
    parser.add_argument("--resubmit_relation", default=None, choices=["combination", "inspiration"],
                        help="Re-process all rows of the given relation type, regardless of their current processed values.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Download the dataset, run sanity checks, and print stats — without calling any LLM.")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("=== DRY RUN — no LLM calls will be made ===")
        logger.info(f"Downloading {HF_FILENAME} from {HF_REPO} …")
        csv_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME, repo_type="dataset")
        df = pd.read_csv(csv_path)
        for _col in ["source_id", "target_id", "publication_year", "paper_id"]:
            if _col in df.columns:
                df[_col] = df[_col].astype(str)
        logger.info(f"Loaded {len(df):,} rows. Columns: {df.columns.tolist()}")

        valid_relations = {"combination", "inspiration"}
        invalid_mask = ~df["relation"].isin(valid_relations)
        n_invalid = invalid_mask.sum()
        n_valid = len(df) - n_invalid
        logger.info(f"Relation stats — valid: {n_valid:,}  |  invalid: {n_invalid:,}  |  total: {len(df):,}")
        for rel, cnt in df["relation"].value_counts().items():
            tag = "" if rel in valid_relations else "  ← INVALID"
            logger.info(f"  {rel!r}: {cnt:,}{tag}")
        if n_invalid > 0:
            logger.warning(f"{n_invalid:,} row(s) with unrecognised relation types would be skipped during processing.")
        else:
            logger.info("All rows have valid relation types.")
        logger.info("=== DRY run complete — exiting. ===")
        return

    secrets = load_secrets(Path(args.secrets))

    # Build only the client(s) needed
    claude_client = None
    openai_client = None
    if _is_claude_model(args.model):
        api_key = secrets.get("anthropic_api_key")
        if not api_key:
            raise ValueError("secrets.yaml is missing 'anthropic_api_key'.")
        claude_client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Using Claude model: {args.model}")
    else:
        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError("secrets.yaml is missing 'openai_api_key'.")
        openai_client = openai.OpenAI(api_key=api_key)
        logger.info(f"Using OpenAI model: {args.model}")

    template = load_prompt_template()
    prompt_template_text = PROMPT_FILE.read_text()
    cost_tracker = CostTracker()

    # --- Download dataset ---
    logger.info(f"Downloading {HF_FILENAME} from {HF_REPO} …")
    csv_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME, repo_type="dataset")
    df = pd.read_csv(csv_path)
    for _col in ["source_id", "target_id", "publication_year", "paper_id"]:
        if _col in df.columns:
            df[_col] = df[_col].astype(str)
    logger.info(f"Loaded {len(df):,} rows. Columns: {df.columns.tolist()}")

    # --- Relation-type sanity check: drop invalid rows before processing ---
    valid_relations = {"combination", "inspiration"}
    invalid_mask = ~df["relation"].isin(valid_relations)
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        counts = df.loc[invalid_mask, "relation"].value_counts()
        logger.warning(
            f"Dropping {n_invalid:,} row(s) with unrecognised relation type:\n"
            + "\n".join(f"  {rel!r}: {cnt:,}" for rel, cnt in counts.items())
        )
        df = df[~invalid_mask].reset_index(drop=True)
        logger.info(f"{len(df):,} rows remaining after filtering.")
    else:
        logger.info("All rows have valid relation types (combination / inspiration).")

    # --- Resume from checkpoint ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        done_df = pd.read_csv(output_path)
        if args.resubmit_failed and not done_df.empty:
            failed_mask = (
                done_df["source_text_processed"].isna() &
                done_df["target_text_processed"].isna()
            )
            n_failed = failed_mask.sum()
            if n_failed > 0:
                logger.info(f"--resubmit_failed: removing {n_failed:,} failed row(s) from checkpoint so they are re-processed.")
                done_df = done_df[~failed_mask].reset_index(drop=True)
        if args.resubmit_relation and not done_df.empty:
            relation_mask = done_df["relation"] == args.resubmit_relation
            n_relation = relation_mask.sum()
            if n_relation > 0:
                logger.info(f"--resubmit_relation: removing {n_relation:,} '{args.resubmit_relation}' row(s) from checkpoint so they are re-processed.")
                done_df = done_df[~relation_mask].reset_index(drop=True)
        done_ids = set(done_df["id"].tolist())
        logger.info(f"Resuming: {len(done_ids):,} rows already processed.")
    else:
        done_df = pd.DataFrame()
        done_ids = set()

    pending = df[~df["id"].isin(done_ids)].copy()
    if args.start_from > 0:
        pending = pending.iloc[args.start_from:]
        logger.info(f"Skipping first {args.start_from} rows (--start_from). Remaining: {len(pending):,}")
    if args.max_rows > 0:
        remaining_budget = max(0, args.max_rows - len(done_ids))
        pending = pending.head(remaining_budget)

    logger.info(f"Rows to process: {len(pending):,}")

    if args.batch:
        debug_entries, example_prompt, example_row_id = run_batch_mode(
            args=args,
            df=df,
            pending=pending,
            template=template,
            claude_client=claude_client,
            openai_client=openai_client,
            output_path=output_path,
            done_df=done_df,
            cost_tracker=cost_tracker,
        )
        write_debug_report(
            args=args,
            prompt_template_text=prompt_template_text,
            example_prompt=example_prompt,
            example_row_id=example_row_id,
            new_debug_entries=debug_entries,
            cost_tracker=cost_tracker,
            total_dataset_rows=len(df),
            processed_rows=len(debug_entries),
            output_path=output_path,
        )
        _log_enrichment_stats(output_path, len(df))
        logger.info("Done.")
        return

    results = []
    debug_entries = []
    example_prompt: Optional[str] = None
    example_row_id: Optional[str] = None

    for i, (idx, row) in enumerate(pending.iterrows()):
        if i % 100 == 0 and i > 0:
            logger.info(f"Progress: {i}/{len(pending)} …")

        try:
            recombination = row_to_recombination(row)
        except ValueError as e:
            logger.warning(f"Skipping row {row['id']}: {e}")
            continue

        prompt = template.render(
            abstract=row["abstract"],
            recombination=recombination,
        )

        if example_prompt is None:
            example_prompt = prompt
            example_row_id = str(row["id"])

        try:
            raw_response = call_llm(
                model=args.model,
                prompt=prompt,
                claude_client=claude_client,
                openai_client=openai_client,
                cost_tracker=cost_tracker,
            )
        except Exception as e:
            logger.error(f"LLM call failed for row {row['id']}: {e}")
            enriched = {"source_text_processed": None, "target_text_processed": None}
        else:
            enriched = parse_enriched(raw_response, row["relation"])

        result_row = row.to_dict()
        result_row.update(enriched)
        results.append(result_row)

        debug_entries.append({
            "id": row["id"],
            "paper_id": row["paper_id"],
            "relation": row["relation"],
            "abstract": row["abstract"],
            "before": recombination["entities"],
            "after": enriched,
        })

        # Checkpoint every 50 rows
        if len(results) % 50 == 0:
            batch_df = pd.DataFrame(results)
            combined = pd.concat([done_df, batch_df], ignore_index=True) if not done_df.empty else batch_df
            _reorder_columns(combined).to_csv(output_path, index=False)
            logger.info(f"  Checkpointed {len(combined):,} rows → {output_path}")
            done_df = combined
            results = []


    # Final save
    if results:
        batch_df = pd.DataFrame(results)
        combined = pd.concat([done_df, batch_df], ignore_index=True) if not done_df.empty else batch_df
        _reorder_columns(combined).to_csv(output_path, index=False)
        logger.info(f"Saved final output: {len(combined):,} rows → {output_path}")
    else:
        logger.info("No new rows to save.")

    # --- Write debug report ---
    write_debug_report(
        args=args,
        prompt_template_text=prompt_template_text,
        example_prompt=example_prompt,
        example_row_id=example_row_id,
        new_debug_entries=debug_entries,
        cost_tracker=cost_tracker,
        total_dataset_rows=len(df),
        processed_rows=len(debug_entries),
        output_path=output_path,
    )

    _log_enrichment_stats(output_path, len(df))
    logger.info("Done.")


if __name__ == "__main__":
    main()
