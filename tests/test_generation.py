import pytest

from rag.generation import ContextBuilder, Generator
from rag.generation.generator import INSUFFICIENT_INFORMATION
from rag.schemas import DocumentChunk, RetrievedChunk


def test_context_builder_formats_sources_and_removes_duplicates() -> None:
    duplicate = RetrievedChunk(
        DocumentChunk("Refunds last 30 days.", {"file_name": "policy.pdf", "page": 4}),
        0.9,
    )
    context = ContextBuilder().build(
        [
            duplicate,
            duplicate,
            RetrievedChunk(
                DocumentChunk("Receipt required.", {"file_name": "terms.md"}),
                0.8,
            ),
        ]
    )

    assert context.count("Refunds last 30 days.") == 1
    assert "[Source 1: policy.pdf, page 4]" in context
    assert "[Source 2: terms.md]" in context


def test_context_builder_honors_character_limit() -> None:
    result = RetrievedChunk(DocumentChunk("x" * 100, {"file_name": "long.txt"}), 1.0)

    context = ContextBuilder(max_characters=40).build([result])

    assert len(context) == 40


def test_generator_passes_grounding_instructions_to_backend() -> None:
    captured: list[dict[str, str]] = []

    def complete(messages: list[dict[str, str]]) -> str:
        captured.extend(messages)
        return "Refunds are available for 30 days. [1]"

    answer = Generator(completion_fn=complete).generate(
        "What is the policy?",
        "[Source 1: policy.pdf]\nRefunds are available for 30 days.",
    )

    assert answer.endswith("[1]")
    assert "only using the provided context" in captured[0]["content"]
    assert "Do not invent facts" in captured[0]["content"]


def test_generator_returns_insufficient_information_without_context() -> None:
    assert Generator().generate("Unknown?", "") == INSUFFICIENT_INFORMATION


def test_offline_generator_uses_grounded_fallback(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    answer = Generator(api_key=None).generate(
        "What is the policy?",
        "[Source 1: policy.pdf]\nRefunds are available for 30 days. More text.",
    )

    assert answer == "Refunds are available for 30 days. [1]"


def test_offline_generator_selects_the_most_relevant_sentence(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    answer = Generator(api_key=None).generate(
        "What must a refund request include?",
        "[Source 1: policy.md]\n"
        "Customers may request a refund within 30 days. "
        "The request must include the order number and a reason.",
    )

    assert answer == "The request must include the order number and a reason. [1]"


def test_explicit_remote_provider_requires_an_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    generator = Generator(api_key=None, provider="openai_compatible")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        generator.generate("Question?", "[Source 1: policy.md]\nEvidence.")
