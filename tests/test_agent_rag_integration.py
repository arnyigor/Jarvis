# tests/test_agent_rag_integration.py
import pytest

from src.agent import AgentBrain
from src.memory import LongTermMemory, WorkingMemory, EpisodicMemory
from src.model_manager import ModelManager
from src.registry import ToolRegistry

@pytest.mark.asyncio
async def test_agent_build_context_includes_rag_results(embedding_model, chromadb_dir):
    # 1. Setup RAG with a known document
    rag = LongTermMemory(chromadb_dir)

    # Add a known document to ChromaDB
    doc_id = "doc1"
    doc_text = "The sky is blue because of Rayleigh scattering."
    embedding = embedding_model.encode([doc_text], normalize_embeddings=True)[0].tolist()
    rag.collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[doc_text]
    )

    # 2. Setup memory stack
    memory_stack = {
        "working": WorkingMemory(max_turns=5, max_tokens=2000),
        "episodic": EpisodicMemory(),
        "rag": rag
    }

    # 3. Simulate user asking about sky color
    memory_stack["working"].push("user", "Why is the sky blue?")

    # 4. Initialize agent (mock model to return empty response)
    class MockModel:
        async def generate(self, prompt):
            return '{"action": null, "params": {}}'  # No tool needed

    model_mgr = ModelManager("dummy")
    model_mgr._client = MockModel()  # Inject mock client

    agent = AgentBrain(
        tool_registry=ToolRegistry(),
        model_mgr=model_mgr,
        memory_stack=memory_stack,
        system_prompt="You are Jarvis."
    )

    # 5. Trigger _build_context()
    context = agent._build_context()

    # 6. Verify RAG result appears in context
    assert "Rayleigh scattering" in context, f"RAG result not found in context:\n{context}"
    assert "The sky is blue because of Rayleigh scattering." in context

@pytest.mark.asyncio
async def test_rag_not_triggered_if_no_recent_user_query(embedding_model, chromadb_dir):
    # Setup: empty working memory
    rag = LongTermMemory(chromadb_dir)
    memory_stack = {
        "working": WorkingMemory(max_turns=5, max_tokens=2000),
        "episodic": EpisodicMemory(),
        "rag": rag
    }

    model_mgr = ModelManager("dummy")
    model_mgr._client = type('Mock', (), {'generate': lambda self, p: '{"action": null}'})()

    agent = AgentBrain(
        tool_registry=ToolRegistry(),
        model_mgr=model_mgr,
        memory_stack=memory_stack,
        system_prompt="You are Jarvis."
    )

    context = agent._build_context()
    assert "Rayleigh scattering" not in context  # No doc added, so safe
