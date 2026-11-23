"""Retrieval evaluation: hit rate, recall@k and MRR over a small labelled question set.

Cases JSON format::

    [{"question": "...", "expected_source": "report.pdf", "expected_keywords": ["revenue"]}]

A question may only be answerable from the top k chunks of one or more documents, so
``expected_sources`` (a list) is accepted as an alternative to ``expected_source``.

A retrieved chunk is *relevant* when its filename matches one of the expected sources
(when given) AND it contains at least one expected keyword (when given).

Metrics (all averaged over the cases):
- **hit rate** -- share of questions with at least one relevant chunk in the top k.
- **recall@k** -- share of the expected sources that were retrieved in the top k.
- **MRR** -- mean reciprocal rank of the first relevant chunk.

Run:  python -m core.evaluation cases.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from core.vector_store import RetrievedChunk


@dataclass
class EvalCase:
    question: str
    expected_source: str | None = None
    expected_keywords: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Keep `expected_source` as a convenient single-document shorthand.
        if self.expected_source and self.expected_source not in self.expected_sources:
            self.expected_sources.insert(0, self.expected_source)

    @property
    def sources(self) -> list[str]:
        return [s.lower() for s in self.expected_sources]


def load_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for row in raw:
        sources = row.get("expected_sources")
        if sources is None:
            sources = [row["expected_source"]] if row.get("expected_source") else []
        cases.append(
            EvalCase(
                row["question"],
                row.get("expected_source"),
                list(row.get("expected_keywords", [])),
                list(sources),
            )
        )
    return cases


def is_relevant(chunk: RetrievedChunk, case: EvalCase) -> bool:
    """True when a chunk matches the case's source and keyword criteria."""
    if case.sources and chunk.filename.lower() not in case.sources:
        return False
    if case.expected_keywords:
        text = chunk.text.lower()
        if not any(keyword.lower() in text for keyword in case.expected_keywords):
            return False
    return bool(case.sources or case.expected_keywords)


# Backwards-compatible name used by older scripts.
def is_hit(chunk: RetrievedChunk, case: EvalCase) -> bool:
    return is_relevant(chunk, case)


def evaluate_retrieval(
    retrieve: Callable[[str], Sequence[RetrievedChunk]],
    cases: Sequence[EvalCase],
    k: int | None = None,
) -> dict:
    """`retrieve(question)` must return ranked chunks. Returns aggregate metrics + per-case detail.

    When `k` is given, only the first `k` chunks are scored (the retriever may return more).
    """
    details: list[dict] = []
    hits = 0
    rr_total = 0.0
    recall_total = 0.0

    for case in cases:
        chunks = list(retrieve(case.question))
        if k is not None:
            chunks = chunks[:k]
        relevant = [chunk for chunk in chunks if is_relevant(chunk, case)]
        rank = next((i for i, chunk in enumerate(chunks, 1) if is_relevant(chunk, case)), None)
        found = {chunk.filename.lower() for chunk in relevant}
        expected = set(case.sources)

        if expected:
            case_recall = len(expected & found) / len(expected)
        else:
            # Keyword-only case: retrieved the relevant chunk or did not.
            case_recall = 1.0 if relevant else 0.0

        hits += rank is not None
        rr_total += 1.0 / rank if rank else 0.0
        recall_total += case_recall
        details.append(
            {
                "question": case.question,
                "first_hit_rank": rank,
                "retrieved": len(chunks),
                "recall": case_recall,
                "found_sources": sorted(found),
            }
        )

    n = max(len(cases), 1)
    return {
        "cases": len(cases),
        "k": k,
        "hit_rate": hits / n,
        "recall_at_k": recall_total / n,
        "mrr": rr_total / n,
        "details": details,
    }


def main(argv: list[str]) -> int:  # pragma: no cover - CLI wrapper
    if len(argv) != 2:
        print("Usage: python -m core.evaluation cases.json")
        return 2
    from config.settings import ensure_directories, load_settings
    from core.embeddings import OllamaEmbedder
    from core.ollama_client import OllamaClient
    from core.retriever import Retriever
    from core.vector_store import VectorStore

    s = load_settings()
    ensure_directories(s)
    retriever = Retriever(VectorStore(s.chroma_dir, s.collection_name), OllamaEmbedder(OllamaClient(s.ollama_base_url)))
    report = evaluate_retrieval(
        lambda q: retriever.retrieve([q], s.embedding_model, s.top_k, s.similarity_threshold, s.search_strategy),
        load_cases(Path(argv[1])),
        k=s.top_k,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
