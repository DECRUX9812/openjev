# Model card — open-Jev v0.1.0

## What this is

A local, CPU-only, open-weights decision layer that answers the same seven questions about a
job posting that the closed hosted model answers. It exists so a firm can run the
"is this a sales lead or an in-house IT requisition?" decision without a metered API, without
the network, and with weights it can read.

It is **not** a reimplementation of the closed model's architecture. It is a distillation of
its *behaviour* onto a frozen open encoder.

## Architecture

| piece | detail | trained? |
|---|---|---|
| Encoder | `BAAI/bge-small-en-v1.5`, 33 M params, quantised ONNX, 384-d | no — frozen |
| Input | `title. Employer: X. Pay: Y` | — |
| Bucket head | MLP 384→384→4, softmax | **yes** |
| Sub-question heads | 5 × MLP 384→384→1, sigmoid | **yes** |
| Fit head | MLP 384→384→5, softmax over the 0–4 fit scale | **yes** |

Head training: cross-entropy against the hosted model's full probability distribution (soft
targets, not argmax), class-balanced weights, rare-bucket oversampling, averaged over
multiple seeds. Runs in minutes on a laptop CPU. No GPU, no LoRA, no fine-tuning of the
encoder.

Frozen-encoder design was chosen deliberately: the shipped head cannot silently rot when
someone upgrades a library, and identical input always gives identical output.

## Data

- **Training:** the production posting stream as labelled by the hosted model. Real job
  postings; **not redistributed with this package**.
- **Evaluation:** 70 hand-labelled postings, stratified (deliberately over-samples the rare
  classes), held out of training entirely.
- **Synthetic postings:** 3,040 generated, 1,500 teacher-labelled, used in experiments and
  **excluded from the shipped model** — they hurt. See
  `docs/synthetic-data-negative-result.md`.

Class balance in the real labelled corpus:

| bucket | rows |
|---|---|
| `generic_job` | 2,579 |
| `staff_role` | 51 |
| `service_lead` | **1** |

## Evaluation

Scored against the 70 hand-labelled postings. The hosted model's own accuracy is computed on
the same 70 rows by comparing its recorded production answers to the hand labels.

| model | gold accuracy | service_lead | staff_role | generic_job |
|---|---|---|---|---|
| hosted Jev (closed) | **68/70 = 97.1%** | 1/2 | 13/14 | 54/54 |
| viral "Jev reproduction" artifact (stock 1.5B + constrained decoding) | 54/70 = 77.1% | 0/2 | 2/14 | 52/54 |
| **open-Jev (this package)** | 66/70 = 94.3% | 0/2 | 12/14 | 54/54 |

**open-Jev does not beat hosted Jev on gold accuracy. It is 2 postings behind (66 vs 68 of 70).**
That is stated plainly here and in `results/openjev_results.json` (`"beats_jev_on_gold": false`)
because it is the honest result, and because the task framing invited the opposite claim.

### Fresh-traffic holdout (independently re-verified)

A separate set of **106 postings that arrived after training** (published in the LM arm's repo at
`verify/live/fresh_106.jsonl`, never seen by any model here) was scored by re-running *this
package's shipped engine* against hosted Jev's recorded answers:

| arm | agreement with hosted Jev on 106 fresh postings |
|---|---|
| **open-Jev classifier (this package)** | **106/106 = 100.0%** |
| open-Jev LM arm (`openjev-lm`) | 104/106 = 98.1% |

That re-run was performed against the shipped `weights/` in this repository, not against a
training-log number, and its predictions matched the LM arm's independently-recorded
`classifier_pred` field on all 106 rows.

### Sister project

The **LM arm** — a LoRA adapter for `Qwen2.5-0.5B-Instruct` trained for 89 minutes on the same
6-vCPU CPU-only host, reaching 65/70 on the gold set — lives at
[`DECRUX9812/openjev-lm`](https://github.com/DECRUX9812/openjev-lm), together with a paper
covering both arms, two independently-written evaluation harnesses, and row-level receipts.

Where open-Jev *does* match the closed model is agreement on real production traffic:

| measure | value |
|---|---|
| bucket agreement with hosted Jev, full production corpus | **2,615 / 2,631 = 99.4%** |
| marginal cost per decision | **$0** |
| latency per decision (CPU, no network) | tens of milliseconds |
| network required | **no** |

The 99.4% figure is agreement with the closed model, not correctness — it says the open model
has absorbed the closed model's behaviour on real traffic. The 94.3% figure is correctness
against human labels. Both are reported; neither substitutes for the other.

`open-Jev` reported numbers come from `results/openjev_results.json` in this repository,
produced by the training run recorded alongside it, with head SHA-256 sums in the same file.
Reproduce with:

```bash
python -m openjev.eval --gold-data <lab>/data --corpus <lab>/runs/leads_corpus_full.json --weights weights
```

## Where open-Jev wins, and where it does not

**Wins, unambiguously:**

- **Cost**: $0 marginal per call, versus a metered API. At the corpus scale in this
  evaluation the hosted model's measured spend is in
  `runs/leads_corpus_full.json`; open-Jev's is zero.
- **Latency**: tens of milliseconds per posting on CPU, no network round trip.
- **Availability**: runs offline, no key, no rate limit, no provider outage.
- **Auditability**: weights are a few hundred KB of open parameters you can read, with
  SHA-256 sums recorded in `results/openjev_results.json`.
- **Determinism**: identical input → identical output, every time.
- **Behavioural agreement**: 99.4% bucket agreement with the closed model across all 2,631
  real production postings.

**Does not win:**

- **Accuracy.** 66/70 versus the closed model's 68/70 on the hand-labelled set. Near-parity,
  not superiority. Anyone claiming an open model *beats* a closed teacher on the teacher's own
  task, while training on the teacher's own labels, should be read with suspicion — the
  ceiling is the teacher, and the teacher's two misses are inherited, not fixable from its
  labels.

## A caveat about the shipped decision bias

The shipped head carries a small per-class decision bias (`[0, 0.75, 0, 0]` over
`service_lead / staff_role / generic_job / junk`) that was tuned on a held-out dev split.

That dev split was drawn to mirror the **gold set**, and the gold set is deliberately stratified
— it over-samples the rare buckets so the metric is informative. Production traffic is nothing
like that: 98% of the real corpus is `generic_job`. So the shipped bias is calibrated for the
evaluation, and a model deployed on real traffic should have its bias **re-derived from that
traffic's own class mix**, or set to zero.

This is not hypothetical. The same weights with the bias forced to zero score **67/70** on gold
versus 66/70 with it. That variant is deliberately not shipped and not claimed as the result —
preferring it requires having looked at the gold set first, which is not a valid way to choose a
model. But it is the right starting point for a production deployment, and it is why this
section exists.

Rule of thumb: tune the bias against the distribution you will actually serve, not against the
distribution you are measuring on.

## Known failure modes

1. **`service_lead` is not reliably detectable.** There is exactly one real labelled example
   in the corpus, and it is in the held-out set. The model can resolve it by generalising
   from the other classes, and sometimes does; it is not trustworthy for this class. Treat
   any `service_lead` prediction as a hint requiring human review, not a decision.
2. **`staff_role` under-prediction.** The class is 51/2,631 of the corpus. Class balancing
   and threshold tuning recover most of it but not all.
3. **Stratified evaluation.** The 70-row gold set over-samples rare classes. Accuracy on a
   raw production stream will be *higher* in raw terms but dominated by `generic_job`, which
   is the easy class. Do not quote the gold number as a production accuracy.
4. **Text-only.** Employer reputation, domain, and posting history are not used. If a
   misclassification matters, that context is the human's job.

## Intended use

Assisting a human reviewer triaging job postings, at zero cost, offline, with the reasoning
available for inspection.

## Out of scope

- Any decision with legal, financial, or employment consequences taken without human review.
- Detecting `service_lead` as a primary signal for outbound sales. One training example is
  not a basis for that.
- Any posting language other than the English the encoder was trained on.

## License

MIT. The encoder is `BAAI/bge-small-en-v1.5` under its own license (MIT); this package
downloads it rather than redistributing it.
