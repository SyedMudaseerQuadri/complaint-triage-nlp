"""Pick real misclassified test examples for the write-up (deterministic) and list top confusions."""
import json
from collections import Counter

import pandas as pd

from src.config import PROC_DIR, REPORTS_DIR
from src.preprocess import truncate

N_EXAMPLES = 8


def main():
    d = pd.read_parquet(PROC_DIR / "test_pred_distilbert.parquet")
    b = pd.read_parquet(PROC_DIR / "test_pred_baseline.parquet")
    labels = sorted(d["label"].unique())
    d["pred_label"] = d["pred"].map(dict(enumerate(labels)))
    base = b.set_index("complaint_id")["pred"].map(dict(enumerate(labels)))
    d["baseline_label"] = base.loc[d["complaint_id"]].values
    wrong = d[d["pred_label"] != d["label"]].copy()
    pairs = Counter(zip(wrong["label"], wrong["pred_label"], strict=True)).most_common(10)
    chosen = []
    # one highest-confidence error from each of the most-confused (true -> predicted) pairs
    for (t, p), _ in pairs:
        sub = wrong[(wrong["label"] == t) & (wrong["pred_label"] == p)]
        sub = sub.sort_values("confidence", ascending=False)
        if len(sub):
            chosen.append(sub.iloc[0])
        if len(chosen) == N_EXAMPLES:
            break
    out = {
        "n_test": int(len(d)), "n_errors": int(len(wrong)),
        "top_confusions": [{"true": t, "predicted": p, "count": c} for (t, p), c in pairs],
        "examples": [
            {"complaint_id": str(r["complaint_id"]), "true": r["label"], "distilbert": r["pred_label"],
             "baseline": r["baseline_label"], "confidence": round(float(r["confidence"]), 3),
             "product_field": r["Product"], "narrative": truncate(r["narrative"], 420)}
            for r in chosen
        ],
    }
    (REPORTS_DIR / "error_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"n_errors": out["n_errors"], "top_confusions": out["top_confusions"][:6]}, indent=1))


if __name__ == "__main__":
    main()
