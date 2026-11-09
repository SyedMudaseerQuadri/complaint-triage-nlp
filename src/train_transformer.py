"""Fine-tune DistilBERT on the complaint narratives (plain PyTorch loop, fp16 on CUDA)."""
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import LABELS, MODELS_DIR, PROC_DIR, REPORTS_DIR, SEED
from src.metrics import compute_metrics, merge_metrics_json, plot_confusion
from src.preprocess import clean_text

MODEL_NAME = "distilbert-base-uncased"
MODEL_VERSION = "distilbert-base-uncased-ft-v1"
MAX_LEN = 256
BATCH = 16
EPOCHS = 3
LR = 5e-5


class Encoded(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.ids, self.labels = enc["input_ids"], labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def load(tok, name):
    df = pd.read_parquet(PROC_DIR / f"{name}.parquet")
    enc = tok([clean_text(t) for t in df["narrative"]], truncation=True, max_length=MAX_LEN)
    return df, Encoded(enc, df["label_id"].tolist())


def make_collate(pad_id):
    def fn(batch):
        n = max(len(ids) for ids, _ in batch)
        x = torch.full((len(batch), n), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, (ids, _) in enumerate(batch):
            x[i, : len(ids)] = torch.tensor(ids)
            mask[i, : len(ids)] = 1
        return {"input_ids": x, "attention_mask": mask,
                "labels": torch.tensor([y for _, y in batch])}

    return fn


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    logits = []
    for b in loader:
        b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            logits.append(model(**b).logits.float().cpu())
    return torch.cat(logits).numpy()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tr_df, tr = load(tok, "train")
    va_df, va = load(tok, "val")
    te_df, te = load(tok, "test")
    col = make_collate(tok.pad_token_id)
    g = torch.Generator().manual_seed(SEED)
    tr_dl = DataLoader(tr, batch_size=BATCH, shuffle=True, collate_fn=col, generator=g)
    va_dl = DataLoader(va, batch_size=64, collate_fn=col)
    te_dl = DataLoader(te, batch_size=64, collate_fn=col)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={lab: i for i, lab in enumerate(LABELS)},
    ).to(dev)
    counts = np.bincount(tr_df["label_id"], minlength=len(LABELS))
    w = torch.tensor(len(tr_df) / (len(LABELS) * counts), dtype=torch.float, device=dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * len(tr_dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    best, best_ep, history, t0 = -1.0, 0, [], time.time()
    MODELS_DIR.mkdir(exist_ok=True)
    out_dir = MODELS_DIR / "distilbert"
    for ep in range(1, EPOCHS + 1):
        model.train()
        run = 0.0
        for i, b in enumerate(tr_dl, 1):
            b = {k: v.to(dev) for k, v in b.items()}
            labels = b.pop("labels")
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logits = model(**b).logits
            loss = loss_fn(logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            if i % 200 == 0:
                print(f"ep {ep} step {i}/{len(tr_dl)} loss {run / i:.4f}", flush=True)
        vm = compute_metrics(va_df["label_id"].values, predict(model, va_dl, dev).argmax(1), LABELS)
        history.append({"epoch": ep, "train_loss": run / len(tr_dl),
                        "val_macro_f1": vm["macro_f1"], "val_accuracy": vm["accuracy"]})
        print(history[-1], flush=True)
        if vm["macro_f1"] > best:
            best, best_ep = vm["macro_f1"], ep
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            (out_dir / "labels.json").write_text(
                json.dumps({"labels": LABELS, "model_version": MODEL_VERSION}))
    train_s = time.time() - t0

    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(dev)
    out = {"model_version": MODEL_VERSION, "base_model": MODEL_NAME, "max_len": MAX_LEN,
           "batch_size": BATCH, "epochs": EPOCHS, "lr": LR, "best_epoch": best_ep,
           "device": str(dev),
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
           "train_seconds": round(train_s, 1), "history": history}
    vlog = predict(model, va_dl, dev)
    tlog = predict(model, te_dl, dev)
    out["val"] = compute_metrics(va_df["label_id"].values, vlog.argmax(1), LABELS)
    out["test"] = compute_metrics(te_df["label_id"].values, tlog.argmax(1), LABELS)
    merge_metrics_json(REPORTS_DIR / "metrics.json", "distilbert", out)
    plot_confusion(out["test"]["confusion_matrix"], LABELS,
                   f"DistilBERT fine-tuned (test macro-F1 {out['test']['macro_f1']:.3f})",
                   REPORTS_DIR / "confusion_distilbert.png")
    probs = torch.softmax(torch.tensor(tlog), 1).numpy()
    te_df = te_df.assign(pred=tlog.argmax(1), confidence=probs.max(1))
    te_df.to_parquet(PROC_DIR / "test_pred_distilbert.parquet")
    print(json.dumps({"val": out["val"]["macro_f1"], "test": out["test"]["macro_f1"]}))


if __name__ == "__main__":
    main()
