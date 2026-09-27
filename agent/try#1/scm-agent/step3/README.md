# Step 3 — Tool Calling 기초

LLM은 함수를 직접 실행하지 못한다. "이 함수를 이 인자로 불러줘"라고 요청만 하고, 실행은 우리 코드가 한다.

## 실행
```
../.venv/bin/python step3_1_tool_basic.py  # tool_calls가 응답에 어떻게 담겨오는지 확인
../.venv/bin/python step3_2_tool_full.py   # 실행 → 결과를 다시 넣기 → 최종 답변까지 전체 흐름
../.venv/bin/python step3_3_tool_multi.py  # 도구가 여러 개일 때 모델이 알아서 고르는지
../.venv/bin/python step3_4_final.py       # 여러 질문에 대해 통합 실행
```

## 관찰 포인트
- `step3_1_tool_basic.py`의 `msg.content`는 보통 비어있고 `tool_calls`만 채워진다 — "실행"이 아니라 "요청"이라는 걸 눈으로 확인.
- `step3_2_tool_full.py`에서 `messages`에 tool 결과를 추가하지 않으면 모델이 절대 최종 답을 못 만든다는 점을 직접 지워보고 확인해보면 이해가 빠르다.
- `step3_3_tool_multi.py`의 "오늘 날씨 어때?"처럼 관련 도구가 없는 질문에는 tool_calls 없이 바로 답하는지 확인.

---

## English Summary

An LLM can't execute functions itself. It only requests "call this function with these args" — your code does the actual execution.

**Run**
```
../.venv/bin/python step3_1_tool_basic.py  # see how tool_calls show up in a response
../.venv/bin/python step3_2_tool_full.py   # full flow: execute → feed result back → final answer
../.venv/bin/python step3_3_tool_multi.py  # does the model pick the right tool among several?
../.venv/bin/python step3_4_final.py       # run several questions through the combined flow
```

**What to notice**
- In `step3_1_tool_basic.py`, `msg.content` is usually empty while `tool_calls` is populated — confirms it's a "request," not an "execution."
- In `step3_2_tool_full.py`, try removing the tool result from `messages` — the model can never produce a final answer without it.
- For an unrelated question like "how's the weather?" in `step3_3_tool_multi.py`, the model should answer directly with no tool_calls at all.
