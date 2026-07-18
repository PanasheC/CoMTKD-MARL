"""Summarize final and best metrics from experiment JSONL files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output", default="summary.csv")
    args = parser.parse_args()
    rows = []
    for metrics_file in Path(args.root).rglob("metrics.jsonl"):
        records = [json.loads(line) for line in metrics_file.read_text().splitlines() if line.strip()]
        if not records:
            continue
        best = max(records, key=lambda record: record.get("val_top1", float("-inf")))
        rows.append(
            {
                "run": str(metrics_file.parent),
                "best_epoch": best.get("epoch"),
                "best_top1": best.get("val_top1"),
                "best_top5": best.get("val_top5"),
                "best_ece": best.get("val_ece"),
                "best_brier": best.get("val_brier"),
                "active_teachers": best.get("train_active"),
                "coherence": best.get("train_coherence"),
            }
        )
    output = Path(args.output)
    if rows:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {len(rows)} runs to {output}")


if __name__ == "__main__":
    main()
