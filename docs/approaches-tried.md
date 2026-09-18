# Approaches tried, and what each scored

Every number here is on the same 70 hand-labelled postings, held out of training entirely, with
any tunable chosen on a separate dev split and never on gold. Hosted Jev scores **68/70 = 97.1%**
on the same set; the viral "reproduction" artifact scores 54/70 = 77.1%.

This table exists so the next person does not spend a night re-running dead ends, and so the
shipped result can be read in context rather than as a lucky single number.

## What shipped

| # | approach | gold | notes |
|---|---|---|---|
| — | **frozen bge-small + small MLP head (this package)** | **66/70 = 94.3%** | 12 seeds ensembled, bias tuned on dev |
| 1 | TF-IDF word 1-2 + char 3-5 → linear softmax, unweighted | 56/70 = 80.0% | collapses to the majority class |
| 2 | same, class-balanced (`(N/(K·n))^0.5`) + rare oversample ×5 | 62/70 = 88.6% | the single biggest win was class weighting |
| 3 | **frozen bge-small + MLP head** | **66-67/70 = 94.3-95.7%** | shipped; the bias choice moves ±1 |
| 4 | multi-task: shared trunk + bucket + 5 aux booleans + fit | 66/70 | no gain over single-task; kept as a variant script |

## What did not work

| approach | gold | why it failed |
|---|---|---|
| synthetic postings only (1,500 teacher-labelled) | 57/70 = 81.4% | learns `service_lead` 2/2 but loses everything else |
| real + synthetic (service_lead only) | 39/70 = 55.7% | perfect `staff_role` 14/14, but 30 generic postings flagged as leads |
| real + synthetic (service_lead + junk) | 63/70 | |
| real + synthetic (all rare) | 62/70 | |
| real + synthetic (everything) | 61/70 | monotone: more synthetic, worse |
| **retrieval/kNN (k=1..35), blended with the head** | knn 60, blend 62 | **actively destroys rows the head gets right** — see below |
| bigger encoder — bge-base, 768-d, same split, same dev-tuned protocol | 66/70 | **no gain** — see below |
| logit-space blend of the two encoders | 66/70 | dev tuning chose weight 0.0 on the small model, i.e. it collapsed to bge-base |
| bge-base, raw argmax with the decision bias set to zero | 67/70 observed | **not claimed** — see below |
| **LoRA SFT of Qwen2.5-0.5B (partial epoch, step 120/420)** | **63/70 = 90.0%** | measured — see below |

### On the larger encoder, and on a number that looks better but is not claimed

The obvious next lever after `bge-small` was `bge-base` (768-d, same frozen-encoder recipe, same
train/dev split). It scores **66/70** under the same protocol the shipped model uses — a bias
chosen on dev. So a 4× larger encoder bought nothing on this task.

Two things worth recording, because both are traps:

1. **Its errors are complementary to `bge-small`'s, not better.** `bge-small` gets
   `generic_job` 54/54 and `staff_role` 12/14; `bge-base` gets `staff_role` 13/14 and
   `generic_job` 53/54. That looks like an ensemble opportunity, so it was built — blending the
   two posterior distributions in logit space with the mixing weight and bias chosen on dev. Dev
   tuning chose a weight of **0.0** on the small model: the blend collapses to `bge-base`. No
   gain.

2. **`bge-base` with the decision bias forced to zero scores 67/70 on gold.** It is tempting to
   ship that instead. It is not shipped, and it is not claimed as the result, because the only
   reason to prefer the zero-bias variant over the dev-tuned one is that we looked at gold
   first. A number selected by inspecting the test set is not a result; it is a memorised
   answer. The dev-tuned protocol gives 66, so 66 is what is reported.

   (The same trap caught this work once already: a first version of the blending script applied
   the decision bias to *probabilities* while the engine applies it to *logits*. That silent
   mismatch produced a fake 67/70 with `staff_role` 14/14. It was caught only by re-scoring the
   same weights through the shipped engine, which disagreed. Always score a candidate through
   the exact code path that will serve it.)

### On the retrieval result, because it is counter-intuitive

Nearest-neighbour over the training embeddings was expected to help the rare classes — it is
usually the tool for that. It scored 60/70 alone, and blending it with the head scored **worse
than the head alone** (62 vs 66). The blend specifically broke rows the head had right:
`Network And Server Analyst`, `Senior Analyst, Corporate Portfolio Management`, `Data Modeler`,
`Application Analyst` — all correctly `staff_role` from the head, all pulled to `generic_job` by
retrieval. With 2,579 of 2,631 neighbours being `generic_job`, the nearest neighbours of a
`staff_role` posting are overwhelmingly generic. The head's learned re-weighting beats the
neighbourhood prior; retrieval has no way to know that rare is rare.

Tuning was clean for this one — the first run was invalid because the class-weight vector in the
tuning objective was indexed in the wrong order, which silently optimised for the wrong class.
Fixed and re-run; the corrected numbers are above. Worth knowing: that bug produced a plausible
looking result (25/70) that a less careful reading would have accepted.

### The LoRA track, finally scored

The last open item was that no LoRA adapter had ever produced a number on the held-out set. It
now has one. A snapshot at **step 120 of 420** (training was still live; the full run cannot
finish on this hardware) scores:

| measure | value |
|---|---|
| bucket accuracy on gold | **63/70 = 90.0%** |
| agreement with hosted Jev | 91.4% |
| mean bucket confidence | 0.929 |
| ECE (5-bin) | 0.044 |
| Brier (4-class) | 0.128 |
| boolean-field agreement | 96.6% |
| fit mean absolute error | 0.235 |

Scored with the lab's own `eval_openjev.py` — parallel constrained decoding, schema forced, so
schema validity is 100% by construction. Every miss is the same shape (7 of 7:
`staff_role`/`service_lead` predicted `generic_job`), which is the same rare-class
under-prediction the frozen-encoder model shows.

**So the frozen-encoder route wins, and not narrowly:** 66/70 for a model that trains in minutes
on CPU, versus 90.0% at 120 of 420 steps for a fine-tune that needs roughly ten hours per epoch
on this hardware. Worth stating plainly since the LoRA route is the more intuitively appealing
one — it is the slower route *and* it was behind at the point of comparison.

A later checkpoint may close some of that gap; the run was still training at report time and its
loss curve was still descending. Nothing here says a completed LoRA run would lose.

## The remaining gap, precisely

The shipped model's four misses:

| truth | predicted | posting | closable? |
|---|---|---|---|
| `staff_role` | `generic_job` @ 0.91 | Lead Developer | maybe — in-house technical role that reads generic |
| `staff_role` | `generic_job` @ 0.80 | Manager, Data and Analysis | maybe — same shape |
| `service_lead` | `generic_job` @ 0.58 | Electronic Business (E-Business) Web Site Developer | only with real `service_lead` training rows |
| `service_lead` | `staff_role` @ 0.83 | Web Designer | **no** — hosted Jev predicts `staff_role` here too |

Two of the four are inherited from the teacher: hosted Jev also gets `Web Designer` wrong, and
distillation cannot recover a label the teacher never produced. The practical ceiling for any
distillation of this teacher on this set is therefore about 69/70, and reaching it needs more
real rare-class labels, not a better architecture.

## What would actually move the number

1. **Human-labelled `service_lead` postings.** There is exactly one in the corpus and it is in
   the held-out set. This is the whole ballgame and it is not a modelling problem.
2. **Domain-randomised synthetic generation** — generate *from real posting templates and
   vocabulary* rather than from a spec matrix, so synthetic text sits inside the real manifold.
   The failures above are a text-distribution failure, not a labelling failure, so changing the
   labels will not help. Untried.
3. **A domain-adapted encoder** — continued pretraining of the encoder on the posting corpus
   (not the head) would move every posting into a better-behaved space. Requires GPU time.
