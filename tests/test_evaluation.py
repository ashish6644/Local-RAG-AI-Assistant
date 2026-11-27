import json

import pytest
from fakes import make_chunk

from core.evaluation import EvalCase, evaluate_retrieval, is_relevant, load_cases


def test_load_cases_supports_single_and_multiple_sources(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {"question": "q1", "expected_source": "a.pdf", "expected_keywords": ["alpha"]},
                {"question": "q2", "expected_sources": ["b.pdf", "c.pdf"]},
            ]
        ),
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert cases[0].expected_sources == ["a.pdf"]
    assert cases[0].expected_keywords == ["alpha"]
    assert cases[1].expected_sources == ["b.pdf", "c.pdf"]
    assert cases[1].expected_keywords == []


def test_relevance_requires_matching_source_and_keyword():
    case = EvalCase("q", "report.pdf", ["revenue"])

    assert is_relevant(make_chunk(text="Revenue grew", filename="report.pdf"), case)
    assert not is_relevant(make_chunk(text="Revenue grew", filename="other.pdf"), case)
    assert not is_relevant(make_chunk(text="Costs fell", filename="report.pdf"), case)


def test_hit_rate_mrr_and_recall_are_computed_per_case():
    cases = [
        EvalCase("q1", "a.pdf", ["alpha"]),
        EvalCase("q2", "b.pdf", ["beta"]),
        EvalCase("q3", "c.pdf", ["gamma"]),
    ]
    results = {
        "q1": [make_chunk(text="nothing", filename="x.pdf"), make_chunk(text="alpha", filename="a.pdf")],
        "q2": [make_chunk(text="beta", filename="b.pdf")],
        "q3": [make_chunk(text="nothing", filename="x.pdf")],
    }

    report = evaluate_retrieval(lambda question: results[question], cases)

    assert report["cases"] == 3
    assert report["hit_rate"] == pytest.approx(2 / 3)
    assert report["recall_at_k"] == pytest.approx(2 / 3)
    assert report["mrr"] == pytest.approx((0.5 + 1.0) / 3)
    assert [detail["first_hit_rank"] for detail in report["details"]] == [2, 1, None]


def test_recall_counts_each_expected_source():
    case = EvalCase("q", expected_sources=["a.pdf", "b.pdf"], expected_keywords=["alpha"])
    chunks = [make_chunk(text="alpha", filename="a.pdf")]

    report = evaluate_retrieval(lambda question: chunks, [case])

    assert report["hit_rate"] == 1.0
    assert report["recall_at_k"] == pytest.approx(0.5)
    assert report["details"][0]["found_sources"] == ["a.pdf"]


def test_k_truncates_chunks_before_scoring():
    case = EvalCase("q", "b.pdf", ["beta"])
    chunks = [make_chunk(text="alpha", filename="a.pdf"), make_chunk(text="beta", filename="b.pdf")]

    report = evaluate_retrieval(lambda question: chunks, [case], k=1)

    assert report["k"] == 1
    assert report["hit_rate"] == 0.0
    assert report["mrr"] == 0.0


def test_keyword_only_case_can_be_a_hit():
    case = EvalCase("q", expected_keywords=["alpha"])

    report = evaluate_retrieval(lambda question: [make_chunk(text="alpha", filename="x.pdf")], [case])

    assert report["hit_rate"] == 1.0
    assert report["recall_at_k"] == 1.0
