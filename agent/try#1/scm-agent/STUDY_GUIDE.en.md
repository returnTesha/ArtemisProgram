# AI Agent Engineering Concepts (Steps 1–30)

A concept notebook focused not on code syntax but on **"why we build it this way."** It covers things I actually ran by hand and verified.

---

## The Big Picture: Where Are These 30 Steps Heading?

```
1~4    : The basic physics of calling an LLM (stateless, structured output, tool calling)
5~9    : "Find what you don't know, then answer" — RAG
10~16  : Remembering, thinking in multiple ways, splitting work across agents, showing progress in real time
17~19  : Building agents that don't die (defense, evaluation, observability)
20     : First completion (capstone)
21~29  : Shifting from "how to think" to "how to ship as a service" (MCP, approval, cost, framework comparison, security, deployment)
30     : Second completion (capstone)
```

**One principle runs through everything**: the LLM API is stateless. This single fact repeats from Step 1 to Step 30, just in different forms — resending the conversation history, feeding tool results back in, state flowing between graph nodes: all of it is the same principle, "it only looks like it remembers if you show it everything again each time."

---

## Part 1. The Basic Physics of LLM Calls (Steps 1–2)

**Step 1 — LLM call basics**
The LLM API is stateless. "Remembering" isn't a capability of the model; we simply resend the entire conversation history every time. temperature=0 is deterministic (same input → same output); the higher it goes, the more the output varies. As conversations grow, tokens overflow, so a sliding window that trims the oldest turns is needed — **what gets cut is always the oldest, never the most recent.**

**Step 2 — Structured output**
A prompt like "answer only in JSON" is not enough: (1) the JSON syntax itself occasionally breaks, and (2) even when the syntax is valid, the values can be wrong (e.g., for `"category": "order|stock|delivery|other"` the model answers with several values mashed together). **"Parsing succeeded" and "the value is correct" are different checks** — enforce the format with JSON Schema, and validate the values with Pydantic. A subtler point: **the order in which you declare fields = the order in which the model generates tokens.** Because an LLM produces the next token while looking at the tokens it has already emitted (autoregressive), putting `reason` before `category` makes it "think, then conclude," while putting it after makes it "decide the conclusion first, then rationalize after the fact." Schema design is not just a validation tool; it's a lever for designing the model's order of thought.

---

## Part 2. Where Agents Begin — Tool Calling (Steps 3–4)

**Step 3 — Tool calling basics**
An LLM can never execute a function directly. When you describe functions via `tools` (name/description/parameters), the LLM only returns a **structured proposal (tool_calls)**: "please call this function with these arguments." Actual execution is 100% our code's job. The flow is always two calls: first call (proposal) → our code executes → second call (sees the result and gives the final answer). When there are several tools, "which one to use, or none at all" is decided by the model purely from the `description` text — routing accuracy depends not on code logic but on the quality of the description wording. **An interesting trap**: the mere fact that `tools` is present in the request makes the model narrow itself to "I should only answer within this tool scope" (tool-anchoring) — which is why it refuses unrelated questions even when the system prompt says nothing.

**Step 4 — Agent loop**
One question may not finish with a single tool — you need a loop that calls tools repeatedly across multiple steps, and `max_steps` prevents infinite loops. A practical trap: when it meets an unsupported request (e.g., "find the item with the lowest stock" — there is no list-query capability), the model invents fake arguments, calls the tool, fails, and may even give up on the other half it could have answered → the system prompt must explicitly say how to respond to unsupported requests. **Small local models can collapse non-deterministically in multi-step tool calling** — the same question gives different answers each run, and sometimes the output even leaks into another language (e.g., Chinese). You can't relax just because `finish_reason=="stop"`; in practice, defensive logic that checks whether a tool_call format has leaked into `content` is essential.

---

## Part 3. Extending Knowledge with Retrieval — RAG (Steps 5–9)

**Step 5 — Embeddings**
An embedding turns text into a vector. Cosine similarity is not a choice but a necessity — it falls straight out of the definition of the vector dot product, works in any number of dimensions, and sin/tan can't serve as similarity measures at all (they can't distinguish 0° from 180°, or they diverge at 90°). **An embedding is not a "judgment" but a replay of a result frozen at training time**, so it can't be corrected by a real-time system prompt — the embeddings API has no messages/role structure to begin with. Embeddings see only surface-level word similarity and know nothing of domain causality ("cause of low stock = delivery delay") → fixing that requires a **two-stage structure: gather generous candidates (retrieve), then have the LLM re-judge only that small set (rerank).** However, **if TOP_K is too small, the true answer is dropped in the first-stage filter**, and no matter how smart the second-stage LLM is, it can't rescue it — this is why recall (not missing things) is more fundamental than reranking. An important conceptual correction: what you embed and store in RAG must be the **"information/knowledge (documents)," not "questions"** — comparing questions to each other is a different use, like FAQ/ticket routing.

**Step 6 — RAG basics (hand-built)**
RAG = Retrieve + Augment (insert into the prompt) + Generate. Building it in plain code without a framework makes the skeleton clear. The key safeguard: the instruction "if it's not in the policy, say you don't know" prevents hallucination. **RAG's fundamental limit**: even when retrieval is accurate, the reasoning done with that information (e.g., the numeric comparison "400 < 500") can still be wrong — **"retrieval accuracy" and "reasoning accuracy" are entirely different problems**, and RAG fixes only the former. The latter has to be handled separately by guardrails (Step 17) or evaluation (Step 18).

**Step 7 — Vector DB (ChromaDB)**
Moving from an in-memory list to a real vector DB gives you persistence + large-scale search + metadata filters (`where`). **The real value of the `where` filter is not convenience**: if you "cut with top_k first, then filter in Python," the true matching answer may lie outside the top_k and you can end up with zero results (confirmed by measurement). `where` works in the opposite order (filter first → top_k within that), eliminating this recall trap at the root — the exact same structure as the TOP_K problem in Step 5. Practical tip: if you don't set `hnsw:space`, ChromaDB uses L2 (Euclidean) distance by default, not cosine — but when vectors are already normalized (length = 1), L2 and cosine give mathematically identical rankings (only the scale differs).

**Step 8 — Improving retrieval quality**
BM25 (keyword search) only checks "do the words overlap" and knows nothing of meaning — so it's strong at exact matches like item codes but weak against paraphrases (the exact opposite of embeddings). Hybrid search (a weighted sum of BM25 + embeddings via `alpha`) blends the strengths of both. **An important structural issue**: "narrow down with embeddings → LLM reranks" alone is not enough — it doesn't use BM25 at all, so it can miss documents whose keywords match exactly. The real complete form is chaining **hybrid search (stage 1) → LLM rerank (stage 2)** — "sweep wide with two cheap methods (keyword + meaning), and spend the expensive one (LLM) only on that small set" is the standard practical pattern.

**Step 9 — Combining Agent + RAG**
RAG search is just another tool (`search_policy`) — from the LLM's perspective, "real-time DB lookup" and "document search" are structurally no different, and it decides by looking only at the description. When a tool is actually used, there are exactly 2 LLM calls (first: get only the tool proposal; in between, run the actual search; second: get the real final answer) — if no tool is needed in the first place, it ends after 1 call.

---

## Part 4. Remembering & Thinking in Multiple Ways (Steps 10–11)

**Step 10 — Memory**
Short-term memory: when a conversation grows long, compress the old part into a summary (the key facts must survive). Long-term memory: store past conversations as vectors and pull them back only when relevant (don't force them in when irrelevant). Note that step10_2 keeps the store in an in-memory list, so it is a demo of the retrieval mechanism only; real persistence across sessions needs a database. Small models sometimes drift languages during summarization (Korean → Chinese), so the language must be pinned explicitly in the prompt.

**Step 11 — Reasoning patterns (ReAct vs Plan-and-Execute)**
These are not "pick whichever suits the situation" — they are **two completely different structures.** ReAct works inside one continuous conversation, at each step actually looking at "what I've found so far" and improvising the next action. Plan-and-Execute (1) makes the whole plan before execution, (2) runs each sub-task as a **completely independent new conversation**, and (3) synthesizes at the end. **A failure I reproduced myself**: give Plan-and-Execute a conditional task like "if stock is under 500 kg, check the delivery date," and even though the planning stage writes "check if the condition is met," the execution stage is an independent conversation that knows nothing about the previous step's result, so it ignores the condition and checks unconditionally. → At design time, choose **ReAct if the task has dependencies/conditions between steps, and Plan-and-Execute if the sub-tasks are independent of each other.**

---

## Part 5. Multiple Agents, Expressed as Graphs (Steps 12–14)

**Step 12 — Multi-agent (Supervisor)**
Instead of one agent with all the tools crammed in, split into role-specific experts plus a routing supervisor. Practical lesson: if you ask for routing as "pick what's needed as a list," the model picks almost everything — **asking separately with independent booleans, like "is stock needed?", "is delivery needed?", is far more accurate for small models.**

**Step 13 — Graph orchestration (hand-built)**
The essence of "graph AI" is simple: **state (just a dict) + nodes (functions that take state and return state) + edges (rules that decide the next node).** Use `add_edge` when "the next node is always fixed," and `add_conditional_edge` when "you must look at the state to decide the next node" (equivalent to if/elif). A graph isn't a straight line; it can loop back (cycles) — this is what lets you express tool-call loops and ReAct iterations as a graph. **An important design distinction**: the graph's **structure (which nodes exist and how they connect) is fixed in advance by a human in code**, and the AI only plays the role of "choosing one of the pre-prepared options" at a fork — the AI does not invent the workflow itself on the fly. (Note: `class Graph` is just a blueprint and does not run by itself; things actually happen when you create an instance with `g = Graph()` and call its methods — the for loop in `Graph.run()` is what actually calls node functions like `check`/`halve`.)

**Step 14 — LangGraph**
Re-implement the engine built by hand in Step 13 with the real LangGraph — `StateGraph`/`add_conditional_edges` map almost 1:1 to what we built. A single `create_agent` line replaces the entire tool-calling loop we wrote by hand in Step 4.

---

## Part 6. Autonomy & Real-Time (Steps 15–16)

**Step 15 — Autonomous loop & reflection**
Self-reflection: don't finalize the answer right away; critique it yourself and then improve — "nothing to fix" is also a legitimate outcome. Autonomous loop: given only a goal, the agent decides its next action and when to stop. Since the model may not behave perfectly as instructed (e.g., ending with plain text instead of calling the termination signal itself), a forced-stop safeguard such as `max_steps` is always needed.

**Step 16 — Streaming**
Instead of waiting for completion, receive pieces as soon as tokens are generated — total time is similar, but the "perceived speed" is completely different (the "typing" effect in chatbot UIs). Tool calls also arrive in pieces (name/arguments split across chunks), so you need logic that stitches them together by `index`. Concretely: keep a `{index: slot}` dict; `setdefault` creates an empty slot the first time an index is seen and reuses the existing slot afterward, and `name`/`arguments` are appended with `+=`. Separate slots per index keep parallel tool calls from mixing.

---

## Part 7. Building Agents That Don't Die (Steps 17–19)

**Step 17 — Guardrails & robustness**
Input validation needs two layers: direct commands like "ignore previous instructions" are blocked instantly by rules, but workarounds like "DAN role-play" only get caught once you go to LLM judgment. You also need output filtering (PII masking, blocking system-prompt leakage) and retries (exponential backoff) to survive in production.

**Step 18 — Evaluation**
Free-form answers with no single correct form are auto-graded by using an LLM as a "judge" (pass if the core facts are right even when the wording differs). **Practical lesson**: asking for `reasoning`/`verdict`/`score` all at once produces self-contradictions where reasoning/verdict and score disagree (e.g. the reasoning says correct but the score is 1) — **when you make an LLM do several judgments at once, they can disagree with each other**. So `score` was dropped and only reasoning+verdict remain, which restored consistency. Also, step18_2 includes a deliberately wrong-reference control case, so the healthy result is 4/5, not 5/5.

**Step 19 — Observability**
Automatically record each function call's input/output/duration (tracing) and aggregate latency p50/p95 (metrics). **Looking only at the average misses "requests that are occasionally slow"** — percentiles catch that. Tokens are free on a local model, but the cost formula carries straight over to real use the moment you switch to a paid API.

---

## Part 8. First Completion — Capstone (Step 20)

The first finished agent, combining the pieces from Steps 6–19 (RAG, memory, graph, guardrails, tracing) into **a single graph.** Each question takes a different path through `input_guard → classify → (specialist node) → respond → output_guard`. The files built here are reused as-is in Step 29 (serving) and Step 30 (capstone v2) — **if you understood the graph skeleton properly in Step 13, this step goes much more smoothly.**

---

## Part 9. From "How to Think" to "How to Ship as a Service" (Steps 21–29)

> From here the focus shifts: not the internals of how an agent works, but the problems you meet when turning it into a real backend service.

**Step 21 — MCP basics**
So far, tools were hard-coded in the code as Python dicts. MCP (Model Context Protocol) is a standard that separates tool definition + execution into a separate server process — the client fetches schemas automatically via `list_tools()` and doesn't have to write them by hand. MCP isn't a new way of programming; it's just a way of "exposing" ordinary functions you already knew.

**Step 22 — MCP + Agent integration**
The agent delegates its whole tool list to the MCP server instead of hard-coding it — add one tool to the server and the agent can use the new tool automatically without changing a single line of agent code. This is the core value of MCP.

**Step 23 — Human-in-the-loop**
Lookup tools are easy to undo if called wrongly, but hard-to-undo tools like actually placing an order need human approval before execution. The `input()` approach blocks the process on the spot, whereas LangGraph's `interrupt()` stops execution entirely and lets you resume later (this step's code uses `InMemorySaver`, so it only resumes within the same process — resuming on another server/process would need a persistent checkpointer). You need the latter approach, plus persistent storage, to build something like a real Slack approval bot.

**Step 24 — Prompt caching**
Placing a repeated long prefix (such as the system prompt) at the front reduces recomputation. The rule: **long unchanging content first, the question that changes each time last** — reverse the order and caching has no effect at all. Caching on local models is detection-based and can be unstable, but cloud APIs' `cache_control` is explicit and guaranteed.

**Step 25 — Cost-based model routing**
Easy questions go to a small/cheap model, hard ones to a large/expensive model. **The real problem I hit in practice**: if you ask a small model to judge for itself whether "this question is complex," it answers "complex" almost every time regardless of the question — **small models often can't do "classification/judgment" at all.** So the judgment itself had to be replaced with regex/keyword-based rules.

**Step 26 — Context compression**
As a conversation grows, what really gets large isn't the conversation turns but the tool-call results. Compressing old tool results into a one-line summary keeps the key facts even after the original is gone. (A real bug: `messages` held a mix of dicts we built and objects returned by the SDK, so uniformly accessing `m["role"]` raised an error — the flip side of what step3_2 taught, that "you can put a pydantic object directly into messages.")

**Step 27 — Framework comparison (AutoGen vs hand-built)**
Compare the hand-built supervisor (routing fixed in code) with AutoGen's `SelectorGroupChat` (asks the LLM for the "next speaker" every turn) — **letting the AI handle routing makes things more flexible but less predictable** (results differ from run to run; this isn't a bug but something to observe).

**Step 28 — Security red-teaming**
Break the Step 17 guardrails yourself from the attacker's side — rule-layer (BLOCKLIST) bypasses are expected (the spaced-out "obfuscated spelling" case certainly evades the rule layer). The "role-play jailbreak" and "authority impersonation" cases also pass the rule layer, but whether the LLM layer catches them is something you observe by running it. A pass isn't a bug; it's red-teaming finding a hole. **A guardrail isn't built once and done; it's a repeated cycle of attack → fix → re-verify.**

**Step 29 — FastAPI serving + deployment**
Wrap the Step 20 agent in an HTTP API and make it deployable with Docker — for `POST /chat` the agent logic is reused as-is via `ask()` (separating logic from the serving layer); the exception is `/chat/stream`, which re-builds the answer prompt inline and is not counted in `/metrics`. Includes the practical issue that `localhost` inside Docker doesn't point to the host (`host.docker.internal`). **Extending Step 16's terminal streaming to the browser requires SSE (Server-Sent Events)** — implemented for real as a `/chat/stream` endpoint: only the graph's last node (respond_node, the one that produces the sentence) is switched to `stream=True` to emit token by token. This exposes a fundamental tension: **output_guard can only judge the "complete sentence," but that sentence has already been streamed out token by token** — real-time streaming and "inspect everything before sending" can't both be fully satisfied at once, and in practice you must choose either (a) buffer and send only what passes, or (b) send first and emit an additional correction event on violation.

---

## Part 10. Final Completion — Capstone v2 (Step 30)

Step 20 plus MCP-based tool execution and a human-in-the-loop approval gate — the form closest to a real service. Core lesson: from the MCP server's point of view, `place_order` is just an ordinary tool no different from `check_stock` — **the judgment that "this tool is dangerous" is the responsibility not of the server but of the client that calls it.** To turn this into a real service, replace the `input()` approval with a separate endpoint or the `interrupt()` pattern (no blocking).

---

## Recurring Core Meta-Concepts (Threads Running Through Everything)

1. **Stateless**: the model remembers nothing. "Memory" is always the result of data we resend every time.
2. **Retrieval accuracy ≠ reasoning accuracy**: even when RAG/tools fetch the right information, the calculation/logic done with it can still be wrong. Different problems, different fixes (guardrails/evaluation).
3. **Limits of small local models**: asking them to classify/judge on their own is unreliable (Step 25), they collapse non-deterministically in multi-step work (Step 4), and asking for several judgments at once yields self-contradiction (Step 18). In practice, replace judgment with rules, or split the fields and ask separately.
4. **Cost/difficulty spectrum**: ① if a prompt can do it, that's cheapest → ② supplement with pipelines/structure (RAG, rerank, hybrid search) → ③ fine-tuning (most expensive, rarely used at small-to-mid scale).
5. **Structure (who designs) vs. judgment (who decides)**: the graph's nodes/edges, the list of tools requiring approval, which tools to expose — this "frame" is always fixed in advance by humans in code, and the AI judges only within that frame.
6. **Safeguards are always needed**: `max_steps` (infinite loops), guardrails (input/output), retries (transient failures), red-teaming (repeatedly verifying defenses) — never assume the AI behaves "perfectly as instructed."
