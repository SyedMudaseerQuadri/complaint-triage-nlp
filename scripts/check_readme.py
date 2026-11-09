"""Check that the headline numbers quoted in README.md match reports/metrics.json.

The README marks every quoted metric as `<!--m:path.to.value-->0.812<!--/m-->`. This script
re-reads metrics.json, formats the value to the same number of decimals and fails on any mismatch.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"<!--m:([\w.\-]+)-->(.*?)<!--/m-->")


def lookup(data, path):
    for part in path.split("."):
        data = data[part]
    return data


def main() -> int:
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    summary = json.loads((ROOT / "reports" / "data_summary.json").read_text())
    data = {**metrics, "data": summary}
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    bad, n = [], 0
    for path, shown in PATTERN.findall(text):
        n += 1
        value = lookup(data, path)
        if isinstance(value, float):
            decimals = len(shown.split(".")[1]) if "." in shown else 0
            if shown.endswith("%"):
                ok = f"{value * 100:.{len(shown.rstrip('%').split('.')[1])}f}%" == shown
            else:
                ok = f"{value:.{decimals}f}" == shown
        else:
            ok = shown.replace(",", "") == str(value)
        if not ok:
            bad.append((path, shown, value))
    print(f"checked {n} tagged numbers")
    for b in bad:
        print("MISMATCH", b)
    return 1 if bad or n == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
