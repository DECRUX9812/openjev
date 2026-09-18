"""openjev.model — the learned decision model.

Two pieces:

1. A frozen sentence encoder (`BAAI/bge-small-en-v1.5`, 384-d, quantised ONNX, CPU). It is
   never fine-tuned here. Keeping it frozen is what makes this reproducible and cheap: the
   whole training job is a few thousand rows through a 384-dim vector and a 2-layer head,
   which finishes in seconds on a CPU.
2. Learned heads over those embeddings — one softmax for `bucket`, sigmoid heads for the five
   yes/no questions, and a 5-way softmax for the `fit` score.

Weights live in `weights/` as a single npz plus a config JSON, so the shipped artefact is a
few hundred kilobytes and loadable without torch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights"


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _mlp(W1, b1, W2, b2, X, activation: str) -> np.ndarray:
    H = np.maximum(X @ W1 + b1, 0.0)
    Z = H @ W2 + b2
    return softmax(Z) if activation == "softmax" else sigmoid(Z)


class DecisionModel:
    """Embeddings + heads. Holds no global state; safe to share between calls."""

    def __init__(self, weights_dir: Path | str = DEFAULT_WEIGHTS, cache_dir: str | None = None):
        self.weights_dir = Path(weights_dir)
        self.config = json.loads((self.weights_dir / "config.json").read_text())
        self.buckets: list[str] = self.config["buckets"]
        self.bias = np.asarray(self.config.get("bias", [0.0] * len(self.buckets)), dtype=np.float32)
        self._cache_dir = cache_dir or str(Path.home() / ".cache" / "fastembed")
        self._embedder = None
        z = np.load(self.weights_dir / "bucket_head.npz")
        self._bucket = (z["W1"], z["b1"], z["W2"], z["b2"])
        self._aux: dict[str, tuple] = {}
        aux_path = self.weights_dir / "aux_heads.npz"
        if aux_path.exists():
            za = np.load(aux_path)
            for key in za.files:
                if key.endswith("__W1"):
                    name = key[: -len("__W1")]
                    self._aux[name] = (za[f"{name}__W1"], za[f"{name}__b1"],
                                       za[f"{name}__W2"], za[f"{name}__b2"])

    # -- embedding ---------------------------------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(model_name=self.config["embedder"],
                                            cache_dir=self._cache_dir)
        return self._embedder

    @staticmethod
    def posting_text(posting: dict) -> str:
        """Exactly the three fields hosted Jev is given. Nothing else may enter this string.

        Notably `legacy_category` is deliberately excluded: it is the regex rule Jev exists to
        replace, so feeding it in would be scoring the answer key.
        """
        clean = lambda s: " ".join(str(s or "").split())  # noqa: E731
        return (f"{clean(posting.get('title'))}. Employer: {clean(posting.get('employer'))}. "
                f"Pay: {clean(posting.get('pay'))}")

    def embed(self, texts: list[str]) -> np.ndarray:
        X = np.asarray(list(self.embedder.embed(texts, batch_size=64)), dtype=np.float32)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    # -- inference ---------------------------------------------------------------
    def predict(self, postings: list[dict]) -> list[dict]:
        X = self.embed([self.posting_text(p) for p in postings])
        bprob = _mlp(*self._bucket, X, "softmax")
        bprob = softmax(np.log(bprob + 1e-9) + self.bias)  # re-normalise after the bias shift
        out = []
        for i in range(len(postings)):
            row = {
                "bucket": {"choice": self.buckets[int(bprob[i].argmax())],
                           "type": "categorical",
                           "confidence": float(bprob[i].max()),
                           "probabilities": {b: float(bprob[i][j]) for j, b in enumerate(self.buckets)}},
            }
            for name, W in self._aux.items():
                p = _mlp(*W, X[i:i + 1], "sigmoid")[0] if W[2].shape[1] == 1 else \
                    _mlp(*W, X[i:i + 1], "softmax")[0]
                if W[2].shape[1] == 1:
                    row[name] = {"choice": bool(p[0] >= 0.5), "type": "boolean",
                                 "probabilities": {"true": float(p[0]), "false": float(1 - p[0])}}
                else:
                    row[name] = {"score": int(p.argmax()), "type": "ordinal",
                                 "confidence": float(p.max()),
                                 "probabilities": {str(k): float(p[k]) for k in range(len(p))}}
            out.append(row)
        return out


def load_model(weights_dir: Path | str = DEFAULT_WEIGHTS, cache_dir: str | None = None) -> DecisionModel:
    return DecisionModel(weights_dir=weights_dir, cache_dir=cache_dir)
