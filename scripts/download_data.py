"""Download the CFPB complaints-with-narratives parquet shards (public domain, CC0).

Why not the official bulk CSV (files.consumerfinance.gov/ccdb/complaints.csv.zip)? When I built
this, that file had no narrative column, and the search API did not return narratives either.
The Hugging Face mirror BEE-spoke-data/consumer-finance-complaints (licence cc0-1.0) keeps the
"Consumer complaint narrative" column and has a `has-text` subset with narratives only (~925 MB).

Usage:  python scripts/download_data.py
Files land in data/raw/hf/ (gitignored). Then run: python -m src.prepare_data
"""
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints/resolve/main/has-text"
SHARDS = [f"train-0000{i}-of-00003.parquet" for i in range(3)]
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "hf"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in SHARDS:
        dest = OUT / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"have {name}")
            continue
        print(f"downloading {name} ...")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(f"{BASE}/{name}", tmp)
        tmp.rename(dest)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
