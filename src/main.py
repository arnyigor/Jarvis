# src/main.py
import asyncio
import logging
import sys
from pathlib import Path

from src.config import Config
from src.python_exec import PythonExecTool
from src.agent import AgentBrain
from src.memory import WorkingMemory, EpisodicMemory, LongTermMemory
from src.model_manager import ModelManager
from src.registry import ToolRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

async def main() -> None:
    # ← создаём конфигурацию по умолчанию
    cfg = Config()

    # 1. Model manager (defaults to GPT‑OSS‑20B)
    model_mgr = ModelManager(default_model=cfg.text_llm)

    # 2. Tool registry
    reg = ToolRegistry()
    reg.register(PythonExecTool())

    # 3. Memory stack
    memory_stack = {
        "working": WorkingMemory(max_turns=10),
        "episodic": EpisodicMemory(),
        "rag": LongTermMemory(Path(cfg.chromadb_dir)),
    }

    # 4. Agent brain
    agent = AgentBrain(reg, model_mgr, memory_stack,
                       system_prompt="You are Jarvis – a privacy‑first multimodal assistant.")

    # Simple CLI loop
    while True:
        try:
            user_query = input("\n> ")
            if user_query.lower() in ("exit", "quit"):
                break
            answer = await agent.run(user_query)
            print(f"\nJarvis: {answer}\n")
        except KeyboardInterrupt:
            break
        except Exception as exc:
            logging.exception("Fatal error")
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
