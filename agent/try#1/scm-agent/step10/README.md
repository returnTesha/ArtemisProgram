# Step 10 — 메모리

대화가 길어지면 토큰이 넘친다. 최근 대화는 그대로 두고 오래된 건 요약하는 단기 메모리, 과거 기록을 임베딩으로 검색해 꺼내오는 장기 메모리의 검색 방식을 만든다 (예제의 저장소는 메모리 내 리스트라 프로그램 종료 시 사라지며, 실제 영속화는 디스크/벡터 DB 저장이 필요하다 — step7 참고).

## 실행
```
../.venv/bin/python step10_1_summary_memory.py  # 오래된 대화를 요약으로 압축
../.venv/bin/python step10_2_vector_memory.py   # 과거 기록을 벡터로 저장해뒀다가 유사할 때 꺼내오기
```

## 관찰 포인트
- `step10_1_summary_memory.py`에서 `[메모리 압축됨]` 로그가 뜬 뒤에도 "아까 말한 협력사 이름이 뭐였지?"에 정확히 답하는지 확인 — 압축돼도 핵심 정보는 남아야 한다.
- `step10_2_vector_memory.py`의 "오늘 날씨 어때?"처럼 관련 기록이 없는 질문에는 "떠올린 기록: (없음)"이 뜨는지 확인 — 억지로 관련 없는 과거 기록을 끌어오지 않는 게 중요하다.
- **실전 노트**: qwen2.5:7b가 요약을 시키면 가끔 한국어 대신 중국어를 섞어 쓰는 경우가 있었다. 프롬프트에 "반드시 한국어로만"을 명시하니 해결됐다 — 작은 모델일수록 언어를 명시적으로 고정해야 한다.

---

## English Summary

Long conversations blow the token budget. Build short-term memory (summarize old turns) and long-term memory (shows the retrieval mechanism; the store is an in-memory list lost on exit, so real cross-session persistence would need saving to disk or a vector DB, cf. step7).

**Run**
```
../.venv/bin/python step10_1_summary_memory.py  # compress old turns into a running summary
../.venv/bin/python step10_2_vector_memory.py   # store past records as vectors, recall when relevant
```

**What to notice**
- After a `[메모리 압축됨]` (memory compacted) log in `step10_1_summary_memory.py`, check the bot still correctly answers "what was that supplier's name again?" — compaction must preserve key facts.
- In `step10_2_vector_memory.py`, an unrelated question ("how's the weather?") should recall nothing — irrelevant history shouldn't be forced in.
- **Real bug found**: qwen2.5:7b occasionally slipped into Chinese when summarizing. Adding "respond in Korean only" to the prompt fixed it — smaller models often need the output language pinned explicitly.
