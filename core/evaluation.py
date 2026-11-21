"""Basic retrieval evaluation: hit rate, recall@k and MRR over a small labelled question set.

Cases JSON format: [{"question": "...", "expected_source": "report.pdf", "expected_keywords": ["revenue"]}]
A retrieved chunk is a hit if its filename matches `expected_source` (when given) AND it contains
at least one expected keyword (when given).

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


def load_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(r["question"], r.get("expected_source"), list(r.get("expected_keywords", [])))
        for r in raw
    ]


def is_hit(chunk: RetrievedChunk, case: EvalCase) -> bool:
    if case.expected_source and chunk.filename.lower() != case.expected_source.lower():
        return False
    if case.expected_keywords:
        text = chunk.text.lower()
        return any(k.lower() in text for k in case.expected_keywords)
    return bool(case.expected_source)


def evaluate_retrieval(
    retrieve: Callable[[str], Sequence[RetrievedChunk]], cases: Sequence[EvalCase]
) -> dict:
    """`retrieve(question)` must return ranked chunks. Returns aggregate metrics + per-case detail."""
    details, hits, rr_total = [], 0, 0.0
    for case in cases:
        chunks = list(retrieve(case.question))
        rank = next((i for i, c in enumerate(chunks, 1) if is_hit(c, case)), None)
        hits += rank is not None
        rr_total += 1.0 / rank if rank else 0.0
        details.append({"question": case.question, "first_hit_rank": rank, "retrieved": len(chunks)})
    n = max(len(cases), 1)
    return {"cases": len(cases), "hit_rate": hits / n, "mrr": rr_total / n, "details": details}


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
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
