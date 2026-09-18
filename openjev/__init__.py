"""open-Jev — an open, local, zero-cost reimplementation of the Jev decision layer.

Jev is a closed hosted model that answers seven typed questions about a job posting so a
firm can tell "an in-house IT requisition" apart from "a small business that needs work".
`open-Jev` answers the same seven questions with the same typed schema, on CPU, offline,
with no API key and no per-token cost.

What is actually open here:

* `openjev.model`   — a bge-small-en-v1.5 sentence encoder plus small learned heads, ~33M +
                      0.3M params, runs in tens of milliseconds per posting on a laptop CPU.
* `openjev.engine`  — the decision layer: typed answers, calibrated probabilities, a hard
                      schema guarantee that a call never returns an out-of-vocabulary label.
* `openjev.distill` — the training script. Reproduces the heads from a labelled corpus.
* `openjev.eval`    — the evaluation harness, including the hand-labelled gold comparison.
* `openjev.synth`   — a generator for the synthetic rare-class postings that the real data
                      stream does not contain enough of.

Read MODEL_CARD.md before trusting a number. The short version: on the lab's 70-posting
hand-labelled gold set this model scores within a point of hosted Jev on the bucket question
while costing nothing per call, and the one class it cannot yet match is reported honestly
rather than hidden.
"""

from .model import DecisionModel, load_model
from .engine import OpenJev, Decision

__all__ = ["DecisionModel", "load_model", "OpenJev", "Decision"]
__version__ = "0.1.0"
