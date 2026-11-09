"""Build the labelled sample + stratified train/val/test split from the raw CFPB parquet shards.

Steps: keep complaints received in [DATE_MIN, DATE_MAX] with a narrative -> map Product to a
merged class -> drop short / duplicate narratives -> cap each class at MAX_PER_CLASS (random,
seeded) -> stratified 80/10/10 split. Writes data/processed/{train,val,test}.parquet and
reports/data_summary.json.
"""
import json

import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from src.config import (
    DATE_MAX,
    DATE_MIN,
    LABELS,
    MAX_PER_CLASS,
    MIN_WORDS,
    PROC_DIR,
    PRODUCT_MAP,
    RAW_DIR,
    REPORTS_DIR,
    SEED,
)

COLS = ["Date received", "Product", "Consumer complaint narrative", "Complaint ID"]


def load_raw() -> pd.DataFrame:
    shards = sorted((RAW_DIR / "hf").glob("*.parquet"))
    if not shards:
        raise SystemExit("No shards in data/raw/hf - run: python scripts/download_data.py")
    parts = []
    for s in shards:
        pf = pq.ParquetFile(s)
        for batch in pf.iter_batches(columns=COLS, batch_size=200_000):
            df = batch.to_pandas()
            df = df[(df["Date received"] >= DATE_MIN) & (df["Date received"] <= DATE_MAX)]
            df = df[df["Consumer complaint narrative"].fillna("").str.len() > 0]
            if len(df):
                parts.append(df)
        print("scanned", s.name, "kept so far", sum(len(p) for p in parts), flush=True)
    return pd.concat(parts, ignore_index=True)


def build(df: pd.DataFrame):
    summary = {"window": [DATE_MIN, DATE_MAX], "n_in_window_with_narrative": int(len(df)),
               "date_min_seen": str(df["Date received"].min()),
               "date_max_seen": str(df["Date received"].max())}
    summary["product_counts_in_window"] = {k: int(v) for k, v in df["Product"].value_counts().items()}
    df = df.rename(columns={"Consumer complaint narrative": "narrative", "Complaint ID": "complaint_id"})
    df["label"] = df["Product"].map(PRODUCT_MAP)
    df = df.dropna(subset=["label"])
    summary["n_after_class_mapping"] = int(len(df))
    df = df[df["narrative"].str.split().str.len() >= MIN_WORDS]
    df = df.drop_duplicates(subset="narrative")
    summary["n_after_min_words_dedupe"] = int(len(df))
    summary["class_counts_before_cap"] = {k: int(v) for k, v in df["label"].value_counts().items()}
    df = pd.concat(
        [g.sample(min(len(g), MAX_PER_CLASS), random_state=SEED) for _, g in df.groupby("label")]
    )
    df["label_id"] = df["label"].map({lab: i for i, lab in enumerate(LABELS)})
    df = df[["complaint_id", "Date received", "Product", "label", "label_id", "narrative"]]
    df = df.reset_index(drop=True)
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)
    summary.update(
        n_total=int(len(df)), n_train=len(train), n_val=len(val), n_test=len(test),
        classes=LABELS, n_classes=len(LABELS),
        class_counts_sample={k: int(v) for k, v in df["label"].value_counts().items()},
        product_to_class=PRODUCT_MAP, seed=SEED, max_per_class=MAX_PER_CLASS,
    )
    return train, val, test, summary


def main():
    df = load_raw()
    train, val, test, summary = build(df)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    for n, d in (("train", train), ("val", val), ("test", test)):
        d.reset_index(drop=True).to_parquet(PROC_DIR / f"{n}.parquet")
    (REPORTS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))
    keys = ("n_in_window_with_narrative", "date_min_seen", "date_max_seen", "n_total",
            "n_train", "n_val", "n_test", "class_counts_before_cap", "class_counts_sample")
    print(json.dumps({k: summary[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
