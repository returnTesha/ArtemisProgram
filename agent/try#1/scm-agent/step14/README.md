# Step 14 — LangGraph

Step 13에서 직접 만든 그래프 엔진의 개념을, 실제 LangGraph 라이브러리로 (3분기로 줄인 버전으로) 재구현해본다.

## 실행
```
../.venv/bin/python step14_1_langgraph_basic.py   # step13_2의 축소판(3분기: 재고/납기/기타)을 LangGraph로 — 규정 RAG 분기는 없음
../.venv/bin/python step14_2_langgraph_agent.py   # LangGraph의 prebuilt ReAct agent(create_agent)
```

## 관찰 포인트
- `step14_1_langgraph_basic.py`와 `step13_2_scm_workflow.py`를 나란히 열어두고 비교해보자 — `StateGraph`/`add_conditional_edges`/`END`가 직접 만든 `Graph`/`add_conditional_edge`/`END`와 거의 1:1로 대응된다.
- `step14_2_langgraph_agent.py`의 `create_agent` 한 줄이, step4에서 직접 짰던 "tool_calls 파싱 → 실행 → 재호출" 루프 전체를 대신한다는 걸 실감해보자.
- **실전 노트**: 처음엔 `langgraph.prebuilt.create_react_agent`를 썼는데 실행 시 deprecation 경고가 떴다. 최신 버전에서는 `langchain.agents.create_agent`로 옮겨졌다 — 빠르게 발전하는 생태계라 API가 자주 바뀌니, 공식 문서를 항상 최신 버전 기준으로 확인하는 습관이 중요하다.

---

## English Summary

Re-implement the concepts of the step 13 graph engine (in a reduced 3-branch form) using the real LangGraph library.

**Run**
```
../.venv/bin/python step14_1_langgraph_basic.py   # a reduced 3-branch version of step13_2 (inventory/delivery/other, no policy RAG branch) via LangGraph
../.venv/bin/python step14_2_langgraph_agent.py   # LangGraph's prebuilt ReAct agent (create_agent)
```

**What to notice**
- Open `step14_1_langgraph_basic.py` next to `step13_2_scm_workflow.py` — `StateGraph`/`add_conditional_edges`/`END` map almost 1:1 onto the hand-built `Graph`/`add_conditional_edge`/`END`.
- `create_agent` in `step14_2_langgraph_agent.py` replaces the entire "parse tool_calls → execute → re-call" loop you hand-wrote in step 4.
- **Real bug found**: `langgraph.prebuilt.create_react_agent` printed a deprecation warning at runtime; the current API moved to `langchain.agents.create_agent`. This ecosystem moves fast — always check the latest docs against your installed version.
