# Step 16 — 스트리밍

토큰이 완성될 때까지 기다리지 않고, 생성되는 즉시 조각(chunk)으로 받는다. 챗봇 UI의 "타이핑 중" 효과의 정체.

## 실행
```
../.venv/bin/python step16_1_streaming_chat.py        # stream=False vs True 응답 속도 체감 비교
../.venv/bin/python step16_2_streaming_tool_agent.py  # 스트리밍 중 tool_calls 조립하기
```

## 관찰 포인트
- `step16_1_streaming_chat.py`의 "첫 토큰까지" 시간과 "전체 완료까지" 시간을 비교해보자 — 전체 시간은 비슷해도 체감 속도는 완전히 다르다.
- `step16_2_streaming_tool_agent.py`는 tool_calls도 이름/인자가 조각조각 나눠서 온다 — `index`로 어느 tool_call에 속하는 조각인지 맞춰 이어붙이는 코드(`tool_calls.setdefault`)를 유심히 보자.
- 로컬 Ollama가 느릴 때(이 환경은 CPU 추론이라 느린 편)는 스트리밍의 체감 효과가 더 크게 느껴진다.

---

## English Summary

Instead of waiting for the full response, receive tokens as they're generated. This is the "typing..." effect behind chat UIs.

**Run**
```
../.venv/bin/python step16_1_streaming_chat.py        # feel the difference between stream=False and True
../.venv/bin/python step16_2_streaming_tool_agent.py  # assemble tool_calls while streaming
```

**What to notice**
- Compare "time to first token" against "time to full completion" in `step16_1_streaming_chat.py` — total time is similar, but perceived speed is completely different.
- In `step16_2_streaming_tool_agent.py`, even tool_calls arrive in fragments (name/args split across chunks) — study how `tool_calls.setdefault` reassembles them by `index`.
- On a slow local Ollama setup (CPU inference, as in this environment), streaming's perceived benefit is even more noticeable.
