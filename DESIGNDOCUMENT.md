# Project Jarvis – Design Document
## 1. Purpose & Scope

| Item | Description |
|------|--------------|
| **Goal** | Build a local multimodal assistant (“Project Jarvis”) that can reason, plan, and act on text + vision inputs while using a limited GPU pool. |
| **Audience** | Lead AI Engineer, Backend/DevOps, QA & Product Ops. |
| **Deliverable** | Functional prototype ready for demo; roadmap to v1.0. |
| **Assumptions** | • LM Studio runs locally (OpenAI‑compatible API). <br>• GPU memory ~4 GB. <br>• All models available via LM Studio. |

---

## 2.  High‑Level Architecture

```
┌─────────────────────┐          ┌───────────────────┐
│    User / Voice     │◄──────►│      Agent        │
│   (CLI / Web UI)    │          │ (JarvisAgent)    │
└────────▲───────────┘          └──────▲────────────┘
         │                                 │
         │                                 │
    Whisper (async)                Core Loop (Think‑Plan‑Act‑Observe)
         │                                 │
         ▼                                 ▼
   ┌────────────────────┐      ┌─────────────────────┐
   │ ModelManager (LM)  │◄───►│  Tool Layer          │
   └────────▲───────────┘      ├─────▲───────▲───────┤
            │                   │     │   │       │
            │                   │     │   │       │
            ▼                   ▼     ▼   ▼       ▼
      LM Studio API         WebSearch  ImageAnalyze  PythonExec
```

* **Core Loop** – implements the “Thought → Plan → Action → Observation” cycle.
* **ModelManager** – abstracts switching between text/vision LLMs via OpenAI‑compatible client.
* **Tool Layer** – encapsulates web search, image analysis, python execution with retry / sandboxing logic.
* **Memory** – sliding token window + ChromaDB RAG.
* **Voice Pipeline** – Whisper (async) feeding transcripts to the core loop.

---

## 3.  Module Breakdown & Interfaces

| Module | Key Classes/Functions | Purpose | External Dependencies |
|--------|-----------------------|---------|----------------------|
| **Orchestrator** | `JarvisAgent`<br>`think()`<br>`act()`<br>`observe()` | Drives LLM reasoning and tool calls. | OpenAI SDK, ModelManager, Tool Layer |
| **Model Management** | `ModelManager`<br>`switch_model()`<br>`generate()` | Load/unload LLMs via LM Studio API; inject conversation context when switching. | openai (OpenAI‑Python), logging |
| **Tools** | `WebSearchTool`<br>`ImageAnalyzeTool`<br>`PythonExecTool` | Encapsulate external operations with retry, sandboxing, and output formatting. | requests, bs4, subprocess, tenacity |
| **Memory** | `TokenWindow`<br>ChromaDB client | Maintain 65 k token context; off‑load older chunks to vector store. | tiktoken / transformers, chromadb |
| **Voice Pipeline** | `AsyncWhisperWorker` | Captures audio, streams transcripts to agent. | whisper.cpp / openai-whisper, asyncio |
| **Utilities** | `RetryWrapper`, `JSONSchemaValidator` | Centralised retry & validation helpers. | tenacity, jsonschema |

### 3.1 Detailed Interfaces

```python
# ModelManager
class ModelManager:
    def __init__(self, default_model: str): ...
    def switch_model(self, model_id: str) -> None:   # unload + load logic
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[ToolDefinition]] = None,
        max_tokens: int = 512,
    ) -> openai.types.chat.completions.ChatCompletion

# Tool Definition (used by LangGraph)
@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]   # JSON schema for arguments

# Tool Layer
class WebSearchTool:
    async def call(self, query: str) -> str

class ImageAnalyzeTool:
    async def call(self, image_path: str) -> str

class PythonExecTool:
    async def call(self, code: str) -> Tuple[str, int]   # (stdout, exit_code)

# Orchestrator
class JarvisAgent:
    def __init__(self): ...
    async def run(self, user_query: str) -> str   # returns final answer
```

---

## 4.  Core Loop Flow

1. **Think** – Send current `history` + new user message to LLM (text model).
2. **Plan** – Parse assistant’s reply; if it contains *tool_calls* → transition to **Act**.
3. **Act** – For each tool call:
    - If `look_at_image`, switch to vision model, run the image analyzer, inject result.
    - If other tools, invoke directly via Tool Layer.
4. **Observation** – Append tool responses (as `role="tool"`) to history and continue loop.
5. **Stop** – When assistant’s choice has `finish_reason == "stop"` or a user‑explicit “Answer” keyword is found, output final text.

All steps are fully asynchronous so that the Whisper worker can keep feeding transcripts in parallel.

---

## 5.  Memory & Context Management

| Component | Function | Notes |
|------------|----------|-------|
| **TokenWindow** | Holds up to 65 k tokens. | Uses deque; on overflow, pops oldest chunk and writes to ChromaDB. |
| **ChromaDB Client** | Stores vector embeddings of summaries and raw text chunks. | Retrieval‑augmented generation: before each LLM call, fetch top‑k relevant passages → prepend to history (≤ 15 k tokens). |
| **Deep Research Mode** | When user asks for a report:<br>- Iteratively search, scrape, summarize, store.<br>- Use summarization pipeline (Qwen‑Mini) to reduce token budget. | Keeps conversation concise and avoids flooding the model with raw HTML. |

---

## 6.  Tool Execution & Reliability

| Tool | Implementation | Retry Policy | Safety |
|------|-----------------|--------------|--------|
| `WebSearchTool` | Calls DuckDuckGo API (or local scraper). | `tenacity.wait_exponential(1, max=5)` | Rate‑limit, avoid duplicate queries. |
| `ImageAnalyzeTool` | Sends image path to Vision LLM; uses ModelManager to swap models. | 3 attempts with back‑off. | Image size check; log if too large. |
| `PythonExecTool` | Runs code in a sandboxed `subprocess.Popen`, timeout=5 s. | None (fail fast). | Capture stdout/stderr, exit code, kill on timeout. |

All tools return structured JSON so that the agent can parse results deterministically.

---

## 7.  Deployment & Runtime

| Layer | Environment | Resources | Notes |
|-------|-------------|-----------|-------|
| **LM Studio** | Local Docker container or native binary | GPU: 4 GB VRAM | Expose `/v1` endpoint on `localhost:1234`. |
| **Agent Service** | Python 3.12, uvicorn (for optional HTTP API) | CPU: 2 vCPU | Runs asynchronously; can be scaled horizontally if multiple agents are needed. |
| **ChromaDB** | Local process or container | Disk: 10 GB | Persisted to disk for RAG persistence. |
| **Voice Service** | Whisper.cpp + PyAudio | None (CPU) | Streams transcripts via asyncio queue. |

---

## 8.  Testing & CI

1. **Unit tests** – For each module (`ModelManager`, `ToolLayer`, `TokenWindow`).
2. **Integration tests** – Simulate full cycle with mock LM Studio responses (use `pytest-asyncio` + `respx`).
3. **End‑to‑end test** – Run the agent in a Docker Compose stack; feed a scripted user query; assert final answer contains expected summary.
4. **Static analysis** – `ruff`, `mypy`.
5. **CI pipeline** – GitHub Actions: lint → type check → unit tests → integration tests.

---

## 9.  Road‑Map & Milestones

| Phase | Deliverable | Timeline |
|-------|-------------|----------|
| **MVP (v0.1)** | Core loop + ModelManager + mock tools; manual CLI demo. | Week 1–2 |
| **Vision Swap** | Real vision model integration, dynamic swapping; demo image analysis. | Week 3 |
| **Memory & RAG** | TokenWindow + ChromaDB; simple deep‑research flow. | Week 4 |
| **Voice Pipeline** | Async Whisper worker; real audio demo. | Week 5 |
| **Robustness** | Retry logic, sandboxed execution, safety checks. | Week 6 |
| **Production Build** | Docker compose stack + monitoring (Prometheus), API endpoint. | Week 7 |

---

## 10.  Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU memory exhaustion when loading both LLM and Vision model | Medium | High | Keep only one large model loaded at a time; unload idle ones quickly. |
| Infinite tool‑call loops | Low | Medium | Max iterations per user turn (e.g., 10). Detect repeated same calls. |
| Over‑flooding context with raw HTML | Medium | High | Auto‑summarize >5k token chunks before ingestion. |
| Whisper latency blocking user input | Low | Medium | Run whisper in separate thread/async task; buffer transcripts. |
| LM Studio API changes | Low | Medium | Abstract via `ModelManager`; write unit tests against stubbed responses. |

---

## 11.  Glossary

- **LLM** – Large Language Model (Qwen‑2.5, Llama3).
- **Vision model** – Multi‑modal model that accepts images (`qwen-vl`).
- **ChromaDB** – Vector database for RAG.
- **LangGraph** – Library to express the orchestrator graphically.

---

## 12.  Appendix – Quick Code Skeleton

```python
# core.py (skeleton)

class ModelManager: ...          # as described above
class JarvisAgent: ...            # uses ModelManager, LangGraph
class WebSearchTool: ...
class ImageAnalyzeTool: ...
class PythonExecTool: ...

if __name__ == "__main__":
    agent = JarvisAgent()
    print(agent.run("Show me a picture of my cat."))
```

---

### Next Steps

1. **Kick‑off meeting** – Align on acceptance criteria & success metrics.
2. **Set up repo + CI** – `git init`, create Dockerfile, GitHub Actions config.
3. **Implement ModelManager & Agent skeleton** – run first local CLI test.
4. **Integrate LM Studio** – point to localhost and verify response round‑trip.
5. **Add Vision swap flow** – implement the image analysis mock.

Once Phase 0 is complete we’ll have a working prototype that can be extended with full toolset, RAG, and voice integration as per the roadmap.

---