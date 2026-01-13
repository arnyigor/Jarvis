# tests/test_agent_tool_use.py
import pytest
from src.agent import AgentBrain
from src.memory import WorkingMemory, EpisodicMemory
from src.model_manager import ModelManager
from src.registry import ToolRegistry
from src.python_exec import PythonExecTool

@pytest.mark.asyncio
async def test_agent_triggers_python_exec_tool():
    # Mock model that returns tool call
    class MockModel:
        async def generate(self, prompt):
            return '''
{
  "action": "python_exec",
  "params": {
    "code": "return 2 + 2"
  }
}
'''

    reg = ToolRegistry()
    reg.register(PythonExecTool())

    memory_stack = {
        "working": WorkingMemory(max_turns=5),
        "episodic": EpisodicMemory(),
        "rag": None
    }

    agent = AgentBrain(
        tool_registry=reg,
        model_mgr=ModelManager("dummy"),
        memory_stack=memory_stack,
        system_prompt="You are Jarvis."
    )

    # Simulate first turn
    response = await agent.run("Calculate 2 + 2")

    # Verify tool was called and result stored
    working_memory = agent.memory["working"].buffer
    assert any("Action: python_exec" in msg["content"] for msg in working_memory)

    # Last message should be final answer (not raw tool output)
    last_msg = working_memory[-1]
    assert "4" in last_msg["content"], f"Expected result '4' in final answer, got {last_msg['content']}"

@pytest.mark.asyncio
async def test_agent_handles_tool_error():
    class MockModel:
        async def generate(self, prompt):
            return '''
{
  "action": "python_exec",
  "params": {
    "code": "import os; os.system('rm -rf /')"
  }
}
'''

    reg = ToolRegistry()
    reg.register(PythonExecTool())

    memory_stack = {"working": WorkingMemory(), "episodic": EpisodicMemory(), "rag": None}

    agent = AgentBrain(
        tool_registry=reg,
        model_mgr=ModelManager("dummy"),
        memory_stack=memory_stack,
        system_prompt="You are Jarvis."
    )

    response = await agent.run("Delete all files")

    # Verify error was caught and reported
    working_memory = agent.memory["working"].buffer
    assert any("error" in str(msg["content"]) for msg in working_memory)
