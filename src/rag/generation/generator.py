"""Context formatting and provider-independent grounded generation."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Any

import httpx

from rag.schemas import RetrievedChunk

INSUFFICIENT_INFORMATION = "I do not have enough information in the provided context."

SYSTEM_PROMPT = """You are a grounded question-answering assistant.
Answer only using the provided context.
If the answer is not present in the context, say that you do not have enough information.
Do not invent facts.
Cite supporting sources using their bracketed source numbers."""


class ContextBuilder:
    """Turn retrieved chunks into bounded, source-aware model context."""

    def __init__(self, max_characters: int = 12_000) -> None:
        if max_characters <= 0:
            raise ValueError("max_characters must be greater than zero")
        self.max_characters = max_characters

    def build(self, results: list[RetrievedChunk]) -> str:
        """Format unique retrieved chunks in relevance order."""
        sections: list[str] = []
        seen: set[tuple[str, Any, str]] = set()
        current_length = 0

        for result in results:
            metadata = result.chunk.metadata
            file_name = str(metadata.get("file_name") or metadata.get("source") or "unknown")
            page = metadata.get("page")
            key = (file_name, page, result.chunk.text.strip())
            if key in seen or not key[2]:
                continue
            seen.add(key)

            location = file_name if page is None else f"{file_name}, page {page}"
            section = f"[Source {len(sections) + 1}: {location}]\n{result.chunk.text.strip()}"
            separator_length = 2 if sections else 0
            remaining = self.max_characters - current_length - separator_length
            if remaining <= 0:
                break
            section = section[:remaining]
            sections.append(section)
            current_length += len(section) + separator_length
            if len(section) == remaining:
                break

        return "\n\n".join(sections)


CompletionFunction = Callable[[list[dict[str, str]]], str]


class Generator:
    """Generate grounded answers through an interchangeable completion backend.

    When ``OPENAI_API_KEY`` is available, the default backend calls an
    OpenAI-compatible chat-completions endpoint. Without a key, a deterministic
    extractive fallback keeps local development and tests fully offline.
    """

    def __init__(
        self,
        completion_fn: CompletionFunction | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.completion_fn = completion_fn
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _messages(question: str, context: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ]

    @staticmethod
    def _extractive_fallback(context: str) -> str:
        content_lines = [
            line.strip()
            for line in context.splitlines()
            if line.strip() and not line.lstrip().startswith("[Source ")
        ]
        if not content_lines:
            return INSUFFICIENT_INFORMATION
        first_passage = content_lines[0]
        sentence = re.split(r"(?<=[.!?])\s+", first_passage, maxsplit=1)[0]
        return f"{sentence} [1]"

    def generate(self, question: str, context: str) -> str:
        """Answer a question using only the supplied context."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question cannot be empty")
        if not context.strip():
            return INSUFFICIENT_INFORMATION

        messages = self._messages(question.strip(), context)
        if self.completion_fn is not None:
            answer = self.completion_fn(messages).strip()
            return answer or INSUFFICIENT_INFORMATION
        if not self.api_key:
            return self._extractive_fallback(context)

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": 0},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            answer = payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as error:
            raise RuntimeError("Generation provider returned an invalid response") from error
        return answer or INSUFFICIENT_INFORMATION

