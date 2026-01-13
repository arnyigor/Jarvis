# tests/test_model_manager.py
import pytest
from src.model_manager import ModelManager

def test_model_manager_lazy_client_initialization():
    mm = ModelManager("qwen3-next-80b-a3b-instruct")

    # Client is not created until accessed
    assert mm._client is None

    # First access triggers init
    client1 = mm.client
    assert isinstance(client1, type(mm._client))

    # Second access returns same instance
    client2 = mm.client
    assert client1 is client2

def test_model_manager_updates_client_on_load():
    mm = ModelManager("qwen3-next-80b-a3b-instruct")

    original_client = mm.client  # Trigger init

    # Simulate model switch
    mm.load("qwen3-next-80b-a3b-instruct")  # This would call lms.llm() in reality

    new_client = mm.client
    assert new_client != original_client
    assert new_client.model == "qwen3-next-80b-a3b-instruct"
