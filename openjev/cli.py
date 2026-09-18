"""openjev.cli — tiny command line so the model is usable without writing Python.

    python -m openjev.cli "Assistant Cook" --employer "Regina Restaurant" --pay "18.00 hourly"
    python -m openjev.cli --json "Web Designer" --employer "PrintWest" --weights weights

Pass --weights to point at a different trained head directory.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="openjev", description="Decide a job posting locally.")
    ap.add_argument("title", help="posting title")
    ap.add_argument("--employer", default="", help="employer name")
    ap.add_argument("--pay", default="", help="pay text, e.g. '25.00 hourly'")
    ap.add_argument("--weights", default=None, help="directory holding the trained heads")
    ap.add_argument("--json", action="store_true", help="emit raw JSON instead of a summary")
    a = ap.parse_args(argv)

    from .engine import OpenJev

    jev = OpenJev(weights_dir=a.weights)
    d = jev.decide({"title": a.title, "employer": a.employer, "pay": a.pay})

    if a.json:
        print(json.dumps(d.to_dict(), indent=2, sort_keys=True))
        return 0

    print(f"bucket : {d.bucket}   (confidence {d.bucket_confidence:.3f})")
    for b, p in sorted(d.bucket_probabilities.items(), key=lambda kv: -kv[1]):
        print(f"         {b:<13} {p:.4f}")
    for k, v in (d.answers or {}).items():
        print(f"{k:<7}: {json.dumps(v)[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
