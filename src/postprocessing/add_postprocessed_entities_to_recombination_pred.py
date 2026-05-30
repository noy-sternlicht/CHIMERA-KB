"""
add_postprocessed_entities_to_recombination_pred.py
----------------------------------------------------
Joins the postprocessed entity columns from noystl/CHIMERA
(source_text_processed, target_text_processed) into each split of
noystl/Recombination-Pred and writes the enriched CSVs locally.

The CHIMERA dataset already contains chimera_with_postprocessed_entities.csv,
which is the output of enrich_chimera.py. Rather than re-running the LLM,
we simply join on the shared 'id' column.

Usage:
    python add_postprocessed_entities_to_recombination_pred.py \
        --output_dir  /path/to/output/dir   # default: <repo-root>/output/recombination_pred_enriched
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

HF_CHIMERA_REPO = "noystl/CHIMERA"
HF_PRED_REPO = "noystl/Recombination-Pred"
POSTPROCESSED_FILE = "chimera_with_postprocessed_entities.csv"
SPLITS = ["train", "valid", "test"]

PROCESSED_COLS = ["source_text_processed", "target_text_processed"]
# The two datasets share no common 'id' column, but (paper_id, source_text, target_text)
# uniquely identifies every row in both and has 100% overlap.
JOIN_KEYS = ["paper_id", "source_text", "target_text"]


def _find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for directory in [current, *current.parents]:
        if (directory / ".git").exists():
            return directory
    raise RuntimeError(f"Could not locate repository root (no .git found above {current})")


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)

    if "relation" in cols and "id" in cols:
        cols.remove("relation")
        cols.insert(cols.index("id") + 1, "relation")

    text_group = [c for c in ["source_text_processed", "target_text_processed", "source_text", "target_text"] if c in cols]
    if text_group:
        insert_at = min(cols.index(c) for c in text_group if c in cols)
        for c in text_group:
            cols.remove(c)
        for i, c in enumerate(text_group):
            cols.insert(insert_at + i, c)

    return df[cols]


def main():
    parser = argparse.ArgumentParser(description="Add postprocessed entity columns to Recombination-Pred splits.")
    parser.add_argument(
        "--output_dir",
        default=str(_find_repo_root() / "output" / "recombination_pred_enriched"),
        help="Directory where enriched CSVs will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download the postprocessed CHIMERA file and keep only what we need
    logger.info(f"Downloading {POSTPROCESSED_FILE} from {HF_CHIMERA_REPO} …")
    chimera_path = hf_hub_download(repo_id=HF_CHIMERA_REPO, filename=POSTPROCESSED_FILE, repo_type="dataset")
    chimera_df = pd.read_csv(chimera_path, usecols=JOIN_KEYS + PROCESSED_COLS)
    logger.info(f"Loaded {len(chimera_df):,} rows with postprocessed entities.")

    for split in SPLITS:
        filename = f"{split}.csv"
        logger.info(f"Downloading {filename} from {HF_PRED_REPO} …")
        split_path = hf_hub_download(repo_id=HF_PRED_REPO, filename=filename, repo_type="dataset")
        split_df = pd.read_csv(split_path)
        n_before = len(split_df)

        merged = split_df.merge(chimera_df, on=JOIN_KEYS, how="left")

        n_missing = int(merged[PROCESSED_COLS[0]].isna().sum())
        if n_missing:
            logger.warning(f"  {split}: {n_missing:,} row(s) had no match in postprocessed CHIMERA and will have NaN for processed columns.")
        else:
            logger.info(f"  {split}: all {n_before:,} rows matched successfully.")

        merged = _reorder_columns(merged)

        out_path = output_dir / filename
        merged.to_csv(out_path, index=False)
        logger.info(f"  Written → {out_path}  ({len(merged):,} rows, {len(merged.columns)} columns)")

    logger.info(f"Done. Enriched splits are in {output_dir}")


if __name__ == "__main__":
    main()
