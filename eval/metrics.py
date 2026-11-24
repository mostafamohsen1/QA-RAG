"""Retrieval and answer metrics for the eval set.

Metrics:
- hit_at_k: did any chunk from the expected (doc, page) land in the top-k context?
- mrr: 1 / rank of the first relevant chunk (0 if none)
- context_precision: fraction of retrieved chunks that are relevant (top-n)
- context_recall: fraction of the single labeled chunk unit found in context (1/0)
- answer_correct: expected answer covered by the generated answer (token overlap gate)
- faithful: LLM-as-judge verdict that the answer is supported by the context
"""

import json
import re
from dataclasses import dataclass, field


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def token_overlap(answer: str, expected: str) -> float:
    a, e = _tokens(answer), _tokens(expected)
    if not e:
        return 0.0
    return len(a & e) / len(e)


def is_answer_correct(answer: str, expected: str, threshold: float = 0.6) -> bool:
    return token_overlap(answer, expected) >= threshold


def is_abstain(answer: str) -> bool:
    return "i don't know" in answer.lower()


@dataclass
class Metrics:
    hit_at_k: float = 0.0
    mrr: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_correct: float = 0.0
    abstain_accuracy: float = 0.0
    faithfulness: float = 0.0
    per_item: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {k: round(v, 4) for k, v in self.__dict__.items() if k != "per_item"}


def validate_eval_set(path: str) -> list[dict]:
    entries = json.load(open(path))
    required = {"id", "question", "expected_answer", "kind"}
    ids = set()
    for e in entries:
        missing = required - e.keys()
        if missing:
            raise ValueError(f"{e.get('id', '?')}: missing fields {missing}")
        if e["kind"] == "grounded" and not e.get("expected_doc"):
            raise ValueError(f"{e['id']}: grounded entry needs expected_doc")
        if e["kind"] not in ("grounded", "abstain"):
            raise ValueError(f"{e['id']}: kind must be grounded|abstain")
        if e["id"] in ids:
            raise ValueError(f"duplicate id {e['id']}")
        ids.add(e["id"])
    return entries
