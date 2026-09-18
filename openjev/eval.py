"""openjev.eval — evaluation harness.

Two evaluations matter, and they answer different questions:

1. **`score_gold`** — accuracy against the 70 hand-labelled postings. This is the number to
   quote when comparing against hosted Jev, because it is the only set a human wrote.
2. **`agreement`** — how often this model returns the same bucket as hosted Jev's recorded
   answers on the full production corpus. High agreement with a close-to-ceiling teacher
   means the model has absorbed the behaviour, not just the label marginal.

Both are deliberately reported side by side. Agreement alone can be gamed by copying the
teacher, so it never stands in for the gold number.

Usage:
    python -m openjev.eval --gold-data DIR --corpus runs/leads_corpus_full.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .engine import OpenJev
from .model import load_model


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def score_gold(jev: OpenJev, sample: list[dict], labels: dict[str, str]) -> dict:
    decisions = jev.decide(sample)
    hits, per, errors = 0, Counter(), []
    totals = Counter(labels.values())
    for d, row in zip(decisions, sample):
        truth = labels[row["id"]]
        if d.bucket == truth:
            hits += 1
            per[truth] += 1
        else:
            errors.append({"id": row["id"], "title": row.get("title"),
                           "truth": truth, "predicted": d.bucket,
                           "confidence": round(d.bucket_confidence, 4)})
    return {
        "n": len(sample), "hits": hits, "accuracy": round(hits / len(sample), 4),
        "per_class": {k: f"{per[k]}/{v}" for k, v in totals.items()},
        "errors": errors,
    }


def agreement(jev: OpenJev, corpus: list[dict]) -> dict:
    decisions = jev.decide(corpus)
    same = sum(1 for d, r in zip(decisions, corpus) if d.bucket == r.get("bucket"))
    dist = Counter(d.bucket for d in decisions)
    return {"n": len(corpus), "same_as_jev": same,
            "agreement": round(same / len(corpus), 4),
            "predicted_distribution": dict(dist)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-data", required=True, help="dir containing leads.sample.jsonl + leads.gold.json")
    ap.add_argument("--corpus", default=None, help="optional runs/leads_corpus_full.json")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    gd = Path(args.gold_data)
    sample = _load_jsonl(gd / "leads.sample.jsonl")
    labels = json.loads((gd / "leads.gold.json").read_text())["labels"]
    jev = OpenJev(weights_dir=args.weights) if args.weights else OpenJev()

    report = {"model": load_model(args.weights).config if args.weights else load_model().config}
    report["gold"] = score_gold(jev, sample, labels)
    if args.corpus:
        corpus = json.loads(Path(args.corpus).read_text())["rows"]
        report["agreement_with_hosted_jev"] = agreement(jev, corpus)

    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
