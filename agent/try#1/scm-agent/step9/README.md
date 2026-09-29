# Step 9 — Agent + RAG 결합

RAG 검색도 결국 하나의 tool이다. "지금 검색이 필요한가"를 LLM이 스스로 판단하게 만들고, 기존 실시간 조회 도구와 합친다.

## 실행
```
../.venv/bin/python step9_1_rag_tool.py  # search_policy 하나만 tool로 등록
../.venv/bin/python step9_2_final.py     # check_stock/check_delivery + search_policy 통합
```

## 관찰 포인트
- `step9_2_final.py`의 시스템 프롬프트를 보면 "실시간 데이터는 check_stock/check_delivery로, 규정은 search_policy로" 구분해준다 — 이 구분이 없으면 모델이 엉뚱한 도구를 고른다.
- "웨이퍼 재고가 안전 수량 기준을 넘었는지"처럼 실시간 데이터 + 규정이 둘 다 필요한 질문에서, 두 도구가 순서대로 호출되는 과정을 `[step N]` 로그로 따라가 보자.
- 두 번째 질문(B 협력사 대체 소싱)에서 모델이 `check_delivery`로 실제 지연 일수를 다시 확인하지 않고 규정만으로 답하는 경우가 있다 — 작은 모델이 "이미 아는 정보"를 다시 검증하지 않으려는 경향을 보여주는 실제 사례다.

---

## English Summary

RAG search is just another tool. Let the LLM decide for itself when a search is needed, and combine it with existing real-time lookup tools.

**Run**
```
../.venv/bin/python step9_1_rag_tool.py  # register search_policy as the only tool
../.venv/bin/python step9_2_final.py     # combine check_stock/check_delivery with search_policy
```

**What to notice**
- The system prompt in `step9_2_final.py` explicitly separates "real-time data → check_stock/check_delivery" from "policy → search_policy" — without that split, the model picks the wrong tool.
- For a question needing both ("is wafer stock over the safety threshold?"), follow the `[step N]` logs to see both tools get called in order.
- In the B-supplier question, the model sometimes answers from policy alone without re-verifying the actual delay via `check_delivery` — a real example of a small model not re-checking what it "already knows."
