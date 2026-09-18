# open-Jev

An **open, local, zero-cost reimplementation of the Jev decision layer** for job postings.

Jev is a closed, hosted model that answers seven typed questions about a job posting so a firm
can tell *"is this an in-house IT requisition, or a small business that needs a website?"*
That distinction is the whole ballgame: it is the difference between a posting that is a
recruiting signal and one that is a **sales lead**.

`openjev` answers the same questions, on your own CPU, with no API key, no network, and no
per-call cost.

```python
from openjev import OpenJev

jev = OpenJev()
d = jev.decide({"title": "Assistant Cook", "employer": "Regina Restaurant",
                "pay": "18.00 hourly"})

d.bucket                     # 'generic_job'
d.bucket_probabilities       # {'generic_job': 0.99, 'junk': 0.004, ...}
d.answers                    # the five boolean sub-questions + fit
d.actionable_service_lead    # False
```

Or from the shell:

```bash
python -m openjev.cli "Assistant Cook" --employer "Regina Restaurant" --pay "18.00 hourly"
```

## Why this exists

The original regex rule could not separate the two cases that matter. A closed model solved
it but costs money per call, needs the network, and its weights are not yours. `openjev` is
the same interface with the decision made locally:

| | Jev (closed) | **open-Jev** |
|---|---|---|
| Weights | closed | **open (MIT)** |
| Runs offline | no | **yes** |
| Marginal cost / call | metered | **$0** |
| Weights you can inspect | no | **yes** |
| Accuracy vs hand labels | see below | see below |

## How it works

Two frozen pieces and one small trained piece.

1. **Encoder** — `BAAI/bge-small-en-v1.5` (33 M params, quantised ONNX, CPU). Never
   fine-tuned. Frozen encoders mean the shipped head cannot silently rot, and re-running the
   same posting twice gives byte-identical output.
2. **Document format** — `title. Employer: X. Pay: Y`. Deliberately the *same* minimal
   information the closed model gets, so the comparison is fair. Notably it does **not** get
   the legacy category column, which is the rule being replaced.
3. **Heads** — a small MLP per question, trained on the encoder's frozen embeddings. This is
   the only trained part and it fits in a few hundred KB.

Training the heads on ~2,500 labelled postings takes **minutes on a laptop CPU** — no GPU,
no fine-tuning, no LoRA. That is the point: the expensive part (language understanding) is a
frozen off-the-shelf encoder, and the part that must be *yours* is tiny and auditable.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
python -m openjev.cli "Web Designer" --employer "PrintWest" --pay "28.00 hourly"
```

## Reproduce the numbers

Point it at your labelled data directory (it needs `leads.sample.jsonl` and
`leads.gold.json`):

```bash
pip install -e .
python -m openjev.eval \
    --gold-data /path/to/typesafe-lab/data \
    --corpus    /path/to/typesafe-lab/runs/leads_corpus_full.json \
    --weights   weights
```

It prints gold accuracy, per-class recall, and — separately — agreement with hosted Jev on
the full production corpus. Both are reported because they answer different questions; see
`openjev/eval.py`.

## Honest limitations

- `service_lead` has **one** real labelled example in the entire corpus, and it is in the
  held-out set. That class is therefore not learnable from real data alone, and `openjev`
  does not claim to detect it reliably. See `MODEL_CARD.md`.
- The gold evaluation set is stratified (deliberately over-samples the rare classes), so its
  accuracy is **not** the accuracy you would see on a raw production stream.
- Synthetic postings were generated and tested as extra training data. They helped the rare
  classes in isolation but **hurt** the real-postings decision boundary in every mixture
  tried, so the shipped model does not use them. That negative result is recorded in
  `docs/synthetic-data-negative-result.md`.

## License

MIT. See `LICENSE`.

The training corpus consists of real job postings and is **not** redistributed here; the
recipe and the trained heads are.
