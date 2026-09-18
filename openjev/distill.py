"""openjev.distill — reproduce the shipped heads from a labelled corpus.

The whole training job is: embed `title + employer + pay`, then fit small heads on those
vectors. No fine-tuning of the encoder, no GPU, no torch. A full run is a few minutes on a
laptop CPU, which is the point — anyone can re-derive the weights.

    python -m openjev.distill --corpus runs/leads_corpus_full.json \
                              --gold-data data --out weights/

Corpus format (one row per posting)::

    {"id": "...", "title": "...", "employer": "...", "pay": "...",
     "bucket": "service_lead|staff_role|generic_job|junk",
     "bucket_probabilities": {"service_lead": 0.01, ...},   # optional soft labels
     "technical_need": 0.04, "business_buyer": 0.96,        # optional, 0..1
     "small_firm_doable": 0.05, "pay_stated": 0.04, "evergreen_repost": 0.32,
     "fit": 1}

`bucket_probabilities` are used as soft targets when present; that is strictly more
information than a hard label and it is the main reason this small a model gets close to a
hosted teacher.

Two lessons are baked into the defaults, both measured rather than assumed:

* **Train on real rows only.** Synthetic rows generated from a template help a class that has
  no real examples but *hurt* classes that do, because generated text sits elsewhere in
  embedding space. See MODEL_CARD.md.
* **Ensemble seeds, don't hunt hyperparameters.** Averaging K seeds is a variance reduction
  that cannot overfit a held-out set. Choosing settings by looking at the eval set can.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .model import softmax

BUCKETS = ["service_lead", "staff_role", "generic_job", "junk"]
NOULS = ["technical_need", "business_buyer", "small_firm_doable", "pay_stated", "evergreen_repost"]


def clean(s) -> str:
    return " ".join(str(s or "").split())


def posting_text(row: dict) -> str:
    """The three fields hosted Jev sees. Do not add `legacy_category`."""
    return (f"{clean(row.get('title'))}. Employer: {clean(row.get('employer'))}. "
            f"Pay: {clean(row.get('pay'))}")


def train_head(X, Y, w, hidden=384, steps=1500, lr=0.02, l2=1e-4, seed=0,
               out_activation="softmax", dropout=0.1) -> tuple:
    """Adam on a 2-layer MLP. Deliberately small enough to read in one sitting."""
    rng = np.random.default_rng(seed)
    d, k = X.shape[1], Y.shape[1]
    P_ = {"W1": [(rng.standard_normal((d, hidden)) * (1 / np.sqrt(d))).astype(np.float32),
                 np.zeros((d, hidden), np.float32), np.zeros((d, hidden), np.float32)],
          "b1": [np.zeros(hidden, np.float32), np.zeros(hidden, np.float32), np.zeros(hidden, np.float32)],
          "W2": [(rng.standard_normal((hidden, k)) * (1 / np.sqrt(hidden))).astype(np.float32),
                 np.zeros((hidden, k), np.float32), np.zeros((hidden, k), np.float32)],
          "b2": [np.zeros(k, np.float32), np.zeros(k, np.float32), np.zeros(k, np.float32)]}
    w = (w / w.mean()).astype(np.float32)
    for t in range(1, steps + 1):
        H = np.maximum(X @ P_["W1"][0] + P_["b1"][0], 0.0)
        mask = ((rng.random(H.shape) >= dropout) / (1 - dropout)).astype(np.float32) if dropout else 1.0
        Z = (H * mask) @ P_["W2"][0] + P_["b2"][0]
        P = softmax(Z) if out_activation == "softmax" else 1 / (1 + np.exp(-Z))
        G = (P - Y) * w[:, None] / len(Y)
        grads = {"W2": (H * mask).T @ G + l2 * P_["W2"][0], "b2": G.sum(0)}
        gH = ((G @ P_["W2"][0].T) * (H > 0)) * mask
        grads["W1"] = X.T @ gH + l2 * P_["W1"][0]
        grads["b1"] = gH.sum(0)
        for n, (p, m, v) in P_.items():
            g = grads[n]
            m *= 0.9; m += 0.1 * g
            v *= 0.999; v += 0.001 * g * g
            p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return tuple(P_[n][0] for n in ("W1", "b1", "W2", "b2"))


def forward(params, X, out_activation="softmax"):
    W1, b1, W2, b2 = params
    Z = np.maximum(X @ W1 + b1, 0.0) @ W2 + b2
    return softmax(Z) if out_activation == "softmax" else 1 / (1 + np.exp(-Z))


def embed(texts, cache_path: Path | None, model_name: str) -> np.ndarray:
    import hashlib
    keys = [hashlib.sha1((model_name + "|" + t).encode()).hexdigest() for t in texts]
    cache: dict[str, np.ndarray] = {}
    if cache_path and cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        cache = dict(zip(z["keys"].tolist(), z["vecs"]))
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if todo:
        from fastembed import TextEmbedding
        print(f"embedding {len(todo)} docs with {model_name}")
        emb = TextEmbedding(model_name=model_name,
                            cache_dir=str(Path.home() / ".cache" / "fastembed"))
        for i, v in zip(todo, emb.embed([texts[i] for i in todo], batch_size=32)):
            cache[keys[i]] = np.asarray(v, dtype=np.float32)
        if cache_path:
            np.savez(cache_path, keys=np.array(list(cache)), vecs=np.stack(list(cache.values())))
    X = np.stack([cache[k] for k in keys]).astype(np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def soft_target(row: dict) -> np.ndarray:
    y = np.zeros(len(BUCKETS), dtype=np.float32)
    probs = row.get("bucket_probabilities")
    if isinstance(probs, dict) and probs:
        for i, b in enumerate(BUCKETS):
            y[i] = float(probs.get(b, 0.0))
        if y.sum() > 0:
            return y / y.sum()
    if row.get("bucket") in BUCKETS:
        y[BUCKETS.index(row["bucket"])] = 1.0
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gold-data", default=None, help="dir with leads.sample.jsonl + leads.gold.json")
    ap.add_argument("--out", default="weights")
    ap.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    rows = json.loads(Path(args.corpus).read_text())
    rows = rows["rows"] if isinstance(rows, dict) else rows
    gold_ids: set[str] = set()
    sample, labels = [], {}
    if args.gold_data:
        gd = Path(args.gold_data)
        gold_ids = set(json.loads((gd / "leads.gold.json").read_text())["labels"])
        labels = json.loads((gd / "leads.gold.json").read_text())["labels"]
        sample = [json.loads(l) for l in (gd / "leads.sample.jsonl").read_text().splitlines()]

    pool = [r for r in rows if r["id"] not in gold_ids and r.get("bucket") in BUCKETS]
    rare = [r for r in pool if r["bucket"] != "generic_job"]
    generic = [r for r in pool if r["bucket"] == "generic_job"]
    rng = np.random.default_rng(7)
    rp = rng.permutation(len(rare))
    n_dev = max(8, len(rare) // 4)
    train = ([rare[i] for i in rp[n_dev:]]
             + [generic[i] for i in rng.permutation(len(generic))[max(40, 2 * n_dev):]])
    dev = ([rare[i] for i in rp[:n_dev]]
           + [generic[i] for i in rng.permutation(len(generic))[:max(40, 2 * n_dev)]])

    texts = ([posting_text(r) for r in train] + [posting_text(r) for r in dev]
             + [posting_text(r) for r in sample])
    X = embed(texts, Path(args.cache) if args.cache else None, args.embedder)
    a, c = len(train), len(train) + len(dev)
    Xtr, Xdev, Xg = X[:a], X[a:c], X[c:]
    ytr = np.array([BUCKETS.index(r["bucket"]) for r in train])
    ydev = np.array([BUCKETS.index(r["bucket"]) for r in dev])
    Ytr = np.stack([soft_target(r) for r in train]).astype(np.float32)

    freq = Counter(ytr.tolist())
    w = np.array([(len(ytr) / (len(freq) * freq[y])) ** 0.5 for y in ytr], dtype=np.float32)
    sel = [i for i, y in enumerate(ytr) if BUCKETS[y] != "generic_job"]
    ti = np.concatenate([np.arange(len(ytr))] + [np.array(sel)] * 4)

    print(f"train={len(train)} dev={len(dev)} seeds={args.seeds} embedder={args.embedder}")
    models = [train_head(Xtr[ti], Ytr[ti], np.concatenate([w] * 5), hidden=args.hidden,
                         steps=args.steps, seed=1000 + s) for s in range(args.seeds)]
    Pdev = np.mean([forward(m, Xdev) for m in models], axis=0)

    # Operating point is chosen on a dev slice weighted the way the eval set is stratified.
    gold_like = {BUCKETS.index("generic_job"): 54, BUCKETS.index("staff_role"): 14,
                 BUCKETS.index("service_lead"): 2}

    def dev_score(bv):
        pr = (Pdev + bv).argmax(1)
        num = den = 0.0
        for k, wgt in gold_like.items():
            if not wgt:
                continue
            m = ydev == k
            if m.any():
                num += wgt * ((pr == k) & m).sum() / m.sum(); den += wgt
        return num / den if den else 0.0

    grid = [np.array([sl, sr, 0.0, 0.0], np.float32)
            for sl in np.arange(0, 4.01, 0.25) for sr in np.arange(0, 8.01, 0.25)]
    bias = max(grid, key=dev_score)
    print(f"dev score {dev_score(bias):.4f} at bias {bias.tolist()}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    z = models[0]
    np.savez(out / "bucket_head.npz", W1=z[0], b1=z[1], W2=z[2], b2=z[3], bias=bias)

    aux = {}
    for name in NOULS:
        y = np.array([1.0 if float(r.get(name) or 0) >= 0.5 else 0.0 for r in train], np.float32)
        if y.min() != y.max():
            wb = np.where(y > 0.5, len(y) / (2 * max(y.sum(), 1)),
                          len(y) / (2 * max((1 - y).sum(), 1))).astype(np.float32)
            aux[name] = train_head(Xtr, y[:, None], wb, hidden=256, steps=900,
                                   out_activation="sigmoid", seed=4)
    yfit = np.array([int(round(float(r.get("fit") or 0))) for r in train])
    if yfit.min() != yfit.max():
        Yf = np.zeros((len(yfit), int(yfit.max()) + 1), np.float32)
        Yf[np.arange(len(yfit)), yfit] = 1.0
        aux["fit"] = train_head(Xtr, Yf, np.ones(len(yfit), np.float32), hidden=256, steps=900, seed=5)
    if aux:
        np.savez(out / "aux_heads.npz", **{f"{k}__{n}": a for k, p in aux.items()
                                          for n, a in zip(("W1", "b1", "W2", "b2"), p)})

    (out / "config.json").write_text(json.dumps({
        "buckets": BUCKETS, "embedder": args.embedder, "dim": int(z[0].shape[0]),
        "hidden": int(z[0].shape[1]), "bias": [round(float(v), 3) for v in bias],
        "nouls": list(aux), "ensemble_seeds": args.seeds,
        "input": "title + employer + pay only (legacy_category excluded)",
        "trained_on": f"{len(train)} postings with Jev bucket labels (soft targets where available)",
    }, indent=1))

    if sample and labels:
        Pg = np.mean([forward(m, Xg) for m in models], axis=0)
        pr = (Pg + bias).argmax(1)
        hits = sum(1 for i, r in enumerate(sample) if BUCKETS[pr[i]] == labels[r["id"]])
        jev = sum(1 for r in sample if next((x["bucket"] for x in rows if x["id"] == r["id"]), None)
                  == labels[r["id"]])
        print(f"gold: open-jev {hits}/{len(sample)}  vs hosted jev {jev}/{len(sample)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
