# Negative result: synthetic postings do not transfer to real postings

**Status: recorded, and deliberately NOT used in the shipped model.**
This document exists so nobody repeats the experiment, and so the shipped numbers are not
mistaken for something they are not.

## Hypothesis

Real labelled data is badly imbalanced:

| bucket | rows in real corpus |
|---|---|
| `generic_job` | 2,579 |
| `staff_role` | 51 |
| `service_lead` | **1** |

`service_lead` is the class the whole task is about, and there is one example — which is
itself held out in the evaluation set. So the hypothesis was: generate synthetic postings
covering the rare families, label them with a strong teacher model, and mix them in.

A generator was built that emits postings by family (`trade_service`, `web_digital`,
`care_home`, `restaurant`, ...) with explicit tactics, so the *intended* bucket is known by
construction. 3,040 postings were generated; 1,500 were labelled by a teacher model:

| teacher label | rows |
|---|---|
| `generic_job` | 903 |
| `staff_role` | 374 |
| `junk` | 136 |
| `service_lead` | **87** |

87 examples of the class that real data has 1 of. On paper, exactly the fix needed.

## Result: synthetic-only learns the rare class, then loses everything else

Same encoder, same head, same evaluation. Only the training mixture differs.

| training data | gold accuracy | service_lead | staff_role | generic_job |
|---|---|---|---|---|
| real only | **66/70 = 94.3%** | 0/2 | 12/14 | 54/54 |
| synthetic only (a0.5) | 57/70 = 81.4% | **2/2** | 2/14 | 45/54 |
| real + synth (service_lead only) | 39/70 = 55.7% | 1/2 | **14/14** | 24/54 |
| real + synth (service_lead + junk) | 63/70 = 90.0% | 0/2 | 9/14 | 54/54 |
| real + synth (all rare) | 62/70 = 88.6% | 0/2 | 8/14 | 54/54 |
| real + synth (everything) | 61/70 = 87.1% | 0/2 | 7/14 | 54/54 |

Read the second row carefully: synthetic-only **does** teach `service_lead` — it resolves
2/2 on real held-out postings, which real-only data cannot do at all. So the synthetic
generator is not broken and the teacher is not wrong.

But every mixture that includes synthetic rows is **worse** than real-only, and every
mixture collapses `generic_job` (54/54 → 24/54 in the worst case). The `+service_lead_only`
row is the clearest failure: it fixes `staff_role` to a perfect 14/14 and resolves a real
`service_lead`, and still scores 55.7% because it now mislabels 30 `generic_job` postings as
leads. A model that flags ordinary jobs as sales leads is worse than useless in production.

## Why it fails

Generated postings are not drawn from the same distribution as real postings. They are
cleaner, more template-like, and lexically flatter. A frozen encoder maps them to a region of
embedding space that overlaps only partly with the real region, so a head trained on the
mixture learns a decision boundary that is correct on the synthetic cluster and displaced on
the real one. Adding synthetic rows does not add information about the real boundary; it adds
a second, competing boundary.

The failure is in the *text distribution*, not the labels. Labelling the synthetic postings
with the real closed model instead of a teacher would not repair it — the same distribution
gap would remain.

## What would actually work

1. **More real rare-class examples.** There is no substitute. One real `service_lead` cannot
   be stretched into a working detector by any amount of generation.
2. **Real-postings-only augmentation** — masking, is not applicable here.
3. **Domain-randomised generation**, i.e. generating *from* real posting templates and
   vocabulary rather than from a spec matrix, so the synthetic text sits inside the real
   manifold. This is the most promising direction, and it is untried.
4. **Two-stage cascade** with a hand-audited threshold on the rare class, accepting precision
   loss only where a human reviews. Also untried.

## Consequence for the shipped model

The shipped `openjev` heads are trained on **real labelled postings only**. `service_lead` is
documented as not reliably detectable, which is the honest position given one training
example. See `MODEL_CARD.md`.
