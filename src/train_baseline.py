"""TF-IDF (word + char n-grams) + class-balanced logistic regression baseline."""
import json
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import LABELS, MODELS_DIR, PROC_DIR, REPORTS_DIR, SEED
from src.metrics import compute_metrics, merge_metrics_json, plot_confusion
from src.preprocess import clean_text

MODEL_VERSION = "baseline-tfidf-lr-v1"


def build_pipeline() -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                                 sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                 max_features=200_000, sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=10.0, max_iter=300, class_weight="balanced", solver="saga",
                             random_state=SEED)
    return Pipeline([("tfidf", feats), ("clf", clf)])


def main():
    train = pd.read_parquet(PROC_DIR / "train.parquet")
    val = pd.read_parquet(PROC_DIR / "val.parquet")
    test = pd.read_parquet(PROC_DIR / "test.parquet")
    for df in (train, val, test):
        df["text"] = df["narrative"].map(clean_text)
    t0 = time.time()
    pipe = build_pipeline()
    pipe.fit(train["text"], train["label_id"])
    fit_s = time.time() - t0
    out = {"model_version": MODEL_VERSION, "fit_seconds": round(fit_s, 1)}
    for name, df in (("val", val), ("test", test)):
        pred = pipe.predict(df["text"])
        out[name] = compute_metrics(df["label_id"].values, pred, LABELS)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "labels": LABELS, "model_version": MODEL_VERSION},
                MODELS_DIR / "baseline.joblib", compress=3)
    REPORTS_DIR.mkdir(exist_ok=True)
    merge_metrics_json(REPORTS_DIR / "metrics.json", "baseline", out)
    plot_confusion(out["test"]["confusion_matrix"], LABELS,
                   f"Baseline TF-IDF + LR (test macro-F1 {out['test']['macro_f1']:.3f})",
                   REPORTS_DIR / "confusion_baseline.png")
    test.assign(pred=pipe.predict(test["text"])).to_parquet(PROC_DIR / "test_pred_baseline.parquet")
    print(json.dumps({k: out[k]["macro_f1"] for k in ("val", "test")}, indent=2))


if __name__ == "__main__":
    main()
