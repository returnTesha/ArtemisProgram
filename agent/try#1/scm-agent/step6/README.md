# Step 6 — RAG 기초 (직접 구현)

RAG = Retrieve(검색) → Augment(프롬프트에 끼워넣기) → Generate(답변 생성). 프레임워크나 벡터DB 없이 이 세 단계를 직접 만들어본다.

## 실행
```
../.venv/bin/python step6_1_chunking.py    # 문서를 어떻게 나누느냐(chunk)가 왜 중요한지
../.venv/bin/python step6_2_naive_rag.py   # 메모리 리스트 + 코사인 유사도로 만든 RAG
```

## 관찰 포인트
- `step6_1_chunking.py`에서 글자 수로 자른 chunk가 문장 중간에서 끊기는 걸 직접 눈으로 확인해보자.
- `step6_2_naive_rag.py`의 "회사 주식 배당금은 언제 나와?" 질문에 "규정에서 확인할 수 없습니다"라고 답하는지 확인 — 관련 없는 문서를 억지로 끌어와 답하지 않는 게 RAG의 핵심 안전장치다.
- 400kg 웨이퍼 재고 질문 답변을 유심히 보면, 검색은 정확한데 모델이 숫자 비교를 틀리는 경우가 있다 — "검색이 맞아도 추론이 틀릴 수 있다"는 RAG의 한계를 보여주는 실제 사례다.

---

## English Summary

RAG = Retrieve → Augment → Generate. Build all three steps yourself, with no framework or vector DB.

**Run**
```
../.venv/bin/python step6_1_chunking.py    # why how you split documents (chunking) matters
../.venv/bin/python step6_2_naive_rag.py   # RAG built from a plain list + cosine similarity
```

**What to notice**
- In `step6_1_chunking.py`, watch how length-based chunking cuts sentences mid-way.
- In `step6_2_naive_rag.py`, "when's the stock dividend?" should get "cannot be confirmed from policy" — refusing to force-fit irrelevant docs is RAG's key safeguard.
- Look closely at the 400kg wafer stock answer: retrieval is correct but the model sometimes gets the number comparison wrong — a real example of "correct retrieval, wrong reasoning."
