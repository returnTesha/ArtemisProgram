# SCM Agent Learning Roadmap (Step 1–30)

*(한국어 원본: [ROADMAP.md](./ROADMAP.md))*

Goal: build up "how an AI agent actually works under the hood" from scratch, using the raw OpenAI SDK
and a local Ollama server (`qwen2.5:7b`, `bge-m3`) — no frameworks (LangChain, etc.) — and then, at the
end, re-implement the same thing with a real framework (LangGraph) to compare. The domain stays fixed
throughout: an SCM (purchase order / inventory / delivery) assistant for a semiconductor company.

Every example uses `client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")` to talk to
the local Ollama server as if it were the OpenAI API.

## Done (Step 1–5)

- **Step 1 — LLM call basics**: a single call, multi-turn conversation, the stateless nature of the API
  (why you must resend the whole conversation history every time), temperature.
- **Step 2 — Structured output**: prompt-only JSON requests → failure cases → enforcing a JSON Schema →
  Pydantic validation → how field order (reason-first) affects accuracy.
- **Step 3 — Tool calling basics**: writing function specs, parsing the LLM's `tool_calls`, feeding
  execution results back into the conversation to get a final answer.
- **Step 4 — Agent loop**: calling multiple tools across multiple steps, a `max_steps` safety cap,
  guiding the model on requests it can't fulfill.
- **Step 5 — Embedding**: text → vectors, comparing meaning via cosine similarity (the foundation of RAG).

## New (Step 6–20)

- **Step 6 — RAG basics (hand-rolled)**: chunk documents, hold every embedding in memory, retrieve by
  cosine similarity, and inject the result into the prompt as context. Understand the essence of RAG
  (retrieve → augment → generate) with no framework and no vector DB.
- **Step 7 — Vector DB (ChromaDB)**: store and search in a real vector DB. Why persistence, large document
  counts, and metadata filtering matter compared to the in-memory approach.
- **Step 8 — Improving search quality**: keyword search (BM25) vs. embedding search, combining both into
  hybrid search, and reranking with an LLM.
- **Step 9 — Combining Agent + RAG**: register the RAG search itself as a tool, so the LLM decides on its
  own whether a search is needed right now (integrated with the step4 tools).
- **Step 10 — Memory**: short-term memory that compresses a long conversation into a summary, and
  long-term memory that stores past conversations as vectors and pulls up similar ones later.
- **Step 11 — Reasoning patterns (ReAct / Plan-Execute)**: ReAct, which explicitly repeats
  Thought → Action → Observation, vs. Plan-and-Execute, which plans everything up front before acting.
- **Step 12 — Multi-agent (Supervisor)**: agents split by role (an inventory specialist, a delivery
  specialist, ...) and a supervisor that decides who should handle a given request.
- **Step 13 — Graph orchestration (hand-rolled)**: build a mini graph executor made of nodes, edges, and
  conditional branches to represent the SCM workflow. Understand what "graph AI" actually does, with no
  framework.
- **Step 14 — LangGraph**: reimplement the graph from step 13 using the real LangGraph library, and
  compare it against LangGraph's prebuilt ReAct agent.
- **Step 15 — Autonomous loops & Reflection**: self-reflection, where the model critiques and improves
  its own answer, and an autonomous loop ("loop AI") that decides its own next action and stopping point
  given only a goal.
- **Step 16 — Streaming**: real-time, token-by-token output with `stream=True`, and how to handle tool
  calls mid-stream.
- **Step 17 — Guardrails & robustness**: input validation (blocking attacks/misuse), output filtering, and
  retry/timeout/exception handling to keep an agent from dying in production.
- **Step 18 — Evaluation**: auto-grading answer quality with LLM-as-judge, and an eval pipeline that runs
  many cases at once.
- **Step 19 — Observability**: decorators that log/trace each execution step, and collecting metrics like
  token usage and latency.
- **Step 20 — Final integration project**: an SCM agent that combines RAG + Tools + Memory + graph
  orchestration + guardrails + tracing, all together.

## Backend deep dive (Step 21–30)

Where 1–20 focused on "how an agent works internally," 21–30 are the problems you actually run into when
turning that into a real backend service.

- **Step 21 — MCP basics**: the Model Context Protocol, which splits tool definition and execution into a
  separate process (server). Build a real MCP server and call `list_tools`/`call_tool` directly.
- **Step 22 — MCP + Agent integration**: the agent delegates its entire tool list and execution to an MCP
  server — the agent code works without ever knowing a tool's concrete implementation.
- **Step 23 — Human-in-the-loop**: hard-to-reverse tools (an actual purchase order) require human approval.
  Implements both blocking on the spot with `input()`, and LangGraph's `interrupt()`, which suspends
  execution itself and can resume later (even from a different process).
- **Step 24 — Prompt caching**: the principle of pinning a repeated, long context up front to cut
  recomputation. The difference between a local model's perceived caching and a cloud API's explicit
  `cache_control`.
- **Step 25 — Cost-based model routing**: easy questions go to a small/cheap model, hard ones to a
  large/expensive model. Why letting a small model judge its own difficulty turns out to be unreliable,
  and why rule-based routing is often needed instead.
- **Step 26 — Context compaction**: what actually grows a long conversation isn't the turns themselves but
  tool-call results. Manage the context budget by compressing old tool results into one-line summaries.
- **Step 27 — Framework comparison**: how a hand-rolled supervisor (step 12) and AutoGen's
  `SelectorGroupChat` handle the same task differently — the real difference between routing decided in
  code vs. routing judged by the LLM every time.
- **Step 28 — Security red-teaming**: try to break the step17 guardrails as an attacker would. An iterative
  process of finding and fixing real bypasses — direct commands, role-play jailbreaks, obfuscated notation,
  and more.
- **Step 29 — FastAPI serving + deployment**: wrap the script-based agent in an HTTP API and containerize
  it with a Dockerfile, including the real-world issue of reaching the host's Ollama from inside a
  container (`host.docker.internal`).
- **Step 30 — Final integration project v2**: the final agent, closest to a real production shape —
  MCP-based tools + human-in-the-loop approval + graph orchestration + guardrails + tracing, all combined.

## How to work through it

Each step is a folder (`stepN/`) containing several small, concept-sized scripts (the same pattern as
steps 1–5). The goal is to run each one yourself, look at the output, and tweak the code until you feel
*why* it behaves the way it does — not just read about it.
