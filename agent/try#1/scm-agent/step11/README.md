# Step 11 — 추론 패턴 (ReAct vs Plan-and-Execute)

ReAct는 한 스텝씩 보고 즉흥적으로 다음 행동을 정한다. Plan-and-Execute는 계획을 전부 세운 뒤 순서대로 실행한다.

## 실행
```
../.venv/bin/python step11_1_react.py          # Thought → Action → Observation 반복
../.venv/bin/python step11_2_plan_execute.py   # 계획 수립 → 순차 실행 → 종합
```

## 관찰 포인트
- `step11_1_react.py`의 `Thought:` 출력을 보면 모델이 "다음에 뭘 확인해야 하는지"를 매 스텝 말로 설명한다 — 디버깅할 때 이게 왜 유용한지 느껴보자.
- `step11_2_plan_execute.py`는 계획(`계획:` 로그)을 먼저 전부 세우고 나서야 실행을 시작한다 — ReAct와 달리 중간에 계획이 바뀌지 않는다.
- **실전 노트**: 초기 버전에서 "B 협력사"라는 이름이 도구 호출 인자로 전달되는 과정에서 "B 협力公司"처럼 한자가 섞여 깨진 적이 있다. 실행 담당 LLM에게 "고유명사를 절대 번역/변형하지 말라"고 명시하니 해결됐다 — 여러 LLM 호출을 거칠수록 고유명사가 새는 지점을 조심해야 한다.

---

## English Summary

ReAct reacts one step at a time, deciding the next action on the fly. Plan-and-Execute plans everything up front, then executes in order.

**Run**
```
../.venv/bin/python step11_1_react.py          # repeating Thought → Action → Observation
../.venv/bin/python step11_2_plan_execute.py   # plan first, then execute in sequence
```

**What to notice**
- The `Thought:` output in `step11_1_react.py` shows the model explaining its next move at every step — notice how useful that is for debugging.
- `step11_2_plan_execute.py` builds the full plan (`계획:` log) before executing anything — unlike ReAct, the plan doesn't change mid-flight.
- **Real bug found**: a supplier name like "B 협력사" got mangled into "B 協力公司" (mixed Hanja) as it passed through a second LLM call. Fixed by explicitly instructing the execution-step LLM to never translate or alter proper nouns — a real risk whenever a value passes through multiple LLM hops.
