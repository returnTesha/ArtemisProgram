# Step 12 — 멀티 에이전트 (Supervisor)

도구를 다 몰아넣은 하나의 에이전트 대신, 역할을 쪼갠 전문가들과 라우팅하는 supervisor를 만든다.

## 실행
```
../.venv/bin/python step12_1_specialists.py  # 재고 전문가 / 납기 전문가 개별 실행
../.venv/bin/python step12_2_supervisor.py   # supervisor가 라우팅 + 결과 종합
```

## 관찰 포인트
- `step12_2_supervisor.py`의 `[supervisor] 라우팅 결정`을 보면 질문마다 어떤 전문가가 호출됐는지 확인할 수 있다.
- "오늘 점심 뭐 먹지?"처럼 SCM과 무관한 질문에는 전문가를 아무도 부르지 않는지 확인.
- **실전 노트**: 처음엔 라우팅을 `needed: list[재고|납기]` 하나의 필드로 물었더니, 모델이 질문 내용과 무관하게 거의 항상 둘 다 골랐다. `needs_inventory`/`needs_delivery`를 독립된 boolean 필드로 나누니 정확해졌다 — 여러 선택지 중 고르게 하는 것보다, "각각 필요한가?"를 따로 물어보는 게 작은 모델엔 더 명확하다.

---

## English Summary

Instead of one agent holding every tool, split responsibilities into specialists and add a supervisor that routes between them.

**Run**
```
../.venv/bin/python step12_1_specialists.py  # run the inventory/delivery specialists individually
../.venv/bin/python step12_2_supervisor.py   # supervisor routes and combines results
```

**What to notice**
- The `[supervisor] 라우팅 결정` (routing decision) log shows which specialist(s) got called per question.
- An unrelated question ("what's for lunch?") should call no specialist at all.
- **Real bug found**: routing was first modeled as one field `needed: list[재고|납기]`, and the model almost always picked both regardless of the question. Splitting it into independent `needs_inventory`/`needs_delivery` booleans fixed it — asking "is X needed?" separately works better for small models than asking them to pick from a list.
