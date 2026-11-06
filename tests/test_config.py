import pytest

from rag.config import Settings


def test_generation_provider_is_validated() -> None:
    with pytest.raises(ValueError, match="generation_provider"):
        Settings(generation_provider="unsupported")

