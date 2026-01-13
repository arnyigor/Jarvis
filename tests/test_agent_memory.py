# tests/test_agent_memory.py
import pytest
from src.memory import WorkingMemory
from sentence_transformers import SentenceTransformer

@pytest.fixture
def tokenizer():
    # Use a lightweight tokenizer for testing (same as Mistral-7B)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

def test_working_memory_push_and_get_recent(tokenizer):
    wm = WorkingMemory(max_turns=10, max_tokens=500)  # Limit to ~500 tokens

    # Push short messages
    wm.push("user", "Hello")
    wm.push("assistant", "Hi there!")
    wm.push("user", "What's the capital of France?")
    wm.push("assistant", "Paris.")

    # Now push a very long message that exceeds token limit
    long_text = "The quick brown fox jumps over the lazy dog. " * 100  # ~600 tokens
    wm.push("user", long_text)

    # Get recent within token limit
    messages = wm.get_recent(token_limit=500)

    assert len(messages) == 2  # Only last two should remain (long + assistant)
    assert messages[-1]["content"] == long_text
    assert messages[-2]["content"] == "Paris."

def test_working_memory_truncates_correctly(tokenizer):
    wm = WorkingMemory(max_turns=5, max_tokens=100)

    # Push 3 short messages totaling ~80 tokens
    wm.push("user", "What is AI?")
    wm.push("assistant", "Artificial Intelligence.")
    wm.push("user", "Explain it.")

    messages = wm.get_recent(token_limit=100)

    assert len(messages) == 3  # All fit

    # Add one more that pushes over
    wm.push("assistant", "A very long explanation that definitely exceeds the token limit. " * 20)

    messages = wm.get_recent(token_limit=100)
    assert len(messages) <= 2  # Should drop oldest to stay under limit
