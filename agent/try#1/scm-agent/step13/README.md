# Step 13 — 그래프 오케스트레이션 (직접 구현)

"그래프 AI"(LangGraph 등)가 실제로 하는 일은 단순하다: 상태(dict) + 노드(함수) + 조건부 엣지(분기). 이걸 직접 만든다.

## 실행
```
../.venv/bin/python step13_1_graph_engine.py    # 미니 그래프 엔진 자체 테스트 (Collatz 예제)
../.venv/bin/python step13_2_scm_workflow.py    # SCM 문의 처리를 그래프로 표현
```

## 관찰 포인트
- `step13_1_graph_engine.py`의 `거쳐간 노드:` 출력을 보면 조건부 분기(`add_conditional_edge`)에 따라 같은 노드(`halve`)를 여러 번 되돌아가는 걸 확인할 수 있다.
- `step13_2_scm_workflow.py`에서 질문마다 `경로:`가 다르게 찍힌다 — 분류 결과에 따라 완전히 다른 노드 경로를 타는 걸 직접 확인해보자.
- 이 파일에 있는 `Graph` 클래스는 step20, step30의 최종 캡스톤에서도 그대로 재사용된다 — 여기서 확실히 이해해두면 뒤가 훨씬 편하다.

---

## English Summary

What "graph AI" (LangGraph, etc.) actually does is simple: state (a dict) + nodes (functions) + conditional edges (branching). Build it yourself.

**Run**
```
../.venv/bin/python step13_1_graph_engine.py    # test the mini graph engine itself (Collatz example)
../.venv/bin/python step13_2_scm_workflow.py    # model SCM query handling as a graph
```

**What to notice**
- The `거쳐간 노드:` (visited nodes) output in `step13_1_graph_engine.py` shows the same node (`halve`) being revisited via `add_conditional_edge`.
- In `step13_2_scm_workflow.py`, each question prints a different `경로:` (path) — see how classification sends it down a completely different node path.
- The `Graph` class here gets reused as-is in the step 20 and step 30 capstones — understanding it now pays off later.
