"""Run the RAG pipeline over eval_set.json and report metrics.

Usage:
    python -m eval.run_eval --k 5            # retrieval + answer, LLM judge off
    python -m eval.run_eval --faithful       # also run LLM-as-judge faithfulness
"""

import argparse
import json
import sys

from eval.metrics import Metrics, is_abstain, is_answer_correct, validate_eval_set


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval/eval_set.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--faithful", action="store_true", help="run LLM-as-judge faithfulness")
    args = ap.parse_args()

    entries = validate_eval_set(args.eval_set)

    from app.core import llm, reranker
    from app.core.pipeline import ingest_document
    from app.core.retriever import hybrid_search
    from app.db.database import SessionLocal
    from app.db.models import Document
    from tests.conftest import PAGES

    db = SessionLocal()
    try:
        if not db.query(Document).filter_by(filename="rag_notes.pdf").first():
            from tests.conftest import _wrapped_pdf_bytes

            ingest_document(db, "rag_notes.pdf", _wrapped_pdf_bytes(PAGES))
    except Exception:
        db.close()
        raise


    m = Metrics()
    grounded = [e for e in entries if e["kind"] == "grounded"]
    abstain = [e for e in entries if e["kind"] == "abstain"]
    n_seen = hit_sum = mrr_sum = prec_sum = recall_sum = 0
    correct_sum = abstain_sum = faith_sum = 0
    try:
        for e in entries:
            candidates = hybrid_search(db, e["question"], top_k=args.k)
            top = reranker.rerank(e["question"], candidates, top_n=args.k)

            relevant_hit_rank = 0
            if e["kind"] == "grounded":
                for rank, c in enumerate(top, start=1):
                    if c["filename"] == e["expected_doc"] and (
                            e["expected_page"] is None or c["page_number"] == e["expected_page"]):
                        relevant_hit_rank = rank
                        break
            hit = 1 if relevant_hit_rank else 0
            n_seen += 1
            hit_sum += hit
            mrr_sum += (1 / relevant_hit_rank) if relevant_hit_rank else 0
            prec_sum += (hit / len(top)) if top else 0
            recall_sum += hit  # one labeled relevant chunk per grounded item

            result = llm.generate_answer(e["question"], top)
            if e["kind"] == "abstain":
                abstain_sum += 1 if is_abstain(result["answer"]) else 0
            else:
                correct_sum += 1 if is_answer_correct(result["answer"], e["expected_answer"]) else 0
                if args.faithful:
                    faith_sum += llm.judge_faithfulness(
                        e["question"], result["answer"], llm.build_context(top))

            m.per_item.append({
                "id": e["id"], "hit@k": hit, "rank": relevant_hit_rank,
                "answer": result["answer"][:160],
            })
    finally:
        db.close()

    n_g = len(grounded)
    m.mrr = mrr_sum / n_g
    m.hit_at_k = hit_sum / n_seen
    m.context_precision = prec_sum / n_g
    m.context_recall = recall_sum / n_g
    m.answer_correct = correct_sum / n_g
    m.abstain_accuracy = (abstain_sum / len(abstain)) if abstain else 0.0
    if args.faithful and n_g:
        m.faithfulness = faith_sum / n_g

    print(json.dumps(m.summary(), indent=2))
    print("\nPer-item:")
    for item in m.per_item:
        print(f"  {item['id']}: hit@{args.k}={item['hit@k']} rank={item['rank']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
