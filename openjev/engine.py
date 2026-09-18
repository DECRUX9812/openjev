"""openjev.engine — the decision layer.

This is the part that has to behave like a decision service rather than a text generator.
Three guarantees, in order of importance:

1. **Schema.** A returned label is always a member of the declared vocabulary. There is no
   path that emits a half-formed or invented bucket, because the vocabulary is a supervised
   classification target, not free text.
2. **Calibrated confidence.** Every answer carries a probability distribution, so a caller can
   route on `confidence` instead of guessing from prose.
3. **Cost and latency.** No network, no key, no per-token billing. Inference is one encoder
   pass plus a 384-dim matmul, which is tens of milliseconds per posting on a laptop CPU.

`decide()` accepts a single posting or a list and always returns `Decision` objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import DecisionModel, load_model

BUCKET_QUESTION = "What kind of buyer is this posting, if any?"


@dataclass
class Decision:
    """One posting's typed answer set."""

    posting_id: str | None
    title: str
    bucket: str
    bucket_confidence: float
    bucket_probabilities: dict[str, float]
    answers: dict[str, Any] = field(default_factory=dict)

    @property
    def actionable_service_lead(self) -> bool:
        """The money question: a small firm that could become contract work."""
        return self.bucket == "service_lead"

    def to_dict(self) -> dict:
        return {
            "id": self.posting_id, "title": self.title, "bucket": self.bucket,
            "bucket_confidence": round(self.bucket_confidence, 4),
            "bucket_probabilities": {k: round(v, 4) for k, v in self.bucket_probabilities.items()},
            "answers": self.answers,
        }


class OpenJev:
    """Drop-in local replacement for the hosted decision call.

    >>> jev = OpenJev()
    >>> d = jev.decide({"title": "Assistant Cook", "employer": "Regina Restaurant",
    ...                 "pay": "18.00 hourly"})
    >>> d.bucket
    'generic_job'
    """

    def __init__(self, weights_dir: str | None = None, model: DecisionModel | None = None,
                 service_lead_threshold: float | None = None):
        self.model = model or load_model(weights_dir) if weights_dir else (model or load_model())
        self.service_lead_threshold = service_lead_threshold

    def decide(self, postings: dict | list[dict]) -> Decision | list[Decision]:
        one = isinstance(postings, dict)
        rows = [postings] if one else list(postings)
        if not rows:
            return [] if not one else self._empty()
        raw = self.model.predict(rows)
        out = [self._to_decision(p, r) for p, r in zip(rows, raw)]
        return out[0] if one else out

    def _to_decision(self, posting: dict, raw: dict) -> Decision:
        b = raw["bucket"]
        return Decision(
            posting_id=posting.get("id"), title=str(posting.get("title") or ""),
            bucket=b["choice"], bucket_confidence=b["confidence"],
            bucket_probabilities=b["probabilities"],
            answers={k: v for k, v in raw.items() if k != "bucket"},
        )

    @staticmethod
    def _empty() -> Decision:
        return Decision(None, "", "", 0.0, {})

    # -- batch convenience -------------------------------------------------------
    def triage(self, postings: list[dict], threshold: float | None = None) -> dict:
        """One call for the operator question: what is worth a human's time, and why.

        Returns the service leads first (highest confidence first), then the in-house
        requisitions, then the rest — plus a count of how many were decided below `threshold`,
        which is the honest measure of how much this run still needs a human.
        """
        thr = self.service_lead_threshold if threshold is None else threshold
        decisions = self.decide(postings)
        leads = [d for d in decisions if d.bucket == "service_lead"]
        staff = [d for d in decisions if d.bucket == "staff_role"]
        rest = [d for d in decisions if d.bucket not in ("service_lead", "staff_role")]
        leads.sort(key=lambda d: -d.bucket_confidence)
        uncertain = ([d for d in decisions if d.bucket_confidence < thr] if thr else [])
        return {
            "service_leads": [d.to_dict() for d in leads],
            "staff_roles": [d.to_dict() for d in staff],
            "other": [d.to_dict() for d in rest],
            "counts": {"service_leads": len(leads), "staff_roles": len(staff),
                       "other": len(rest), "total": len(decisions)},
            "below_threshold": len(uncertain),
        }
