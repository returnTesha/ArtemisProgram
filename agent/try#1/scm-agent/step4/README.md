# Step 4 — Agent Loop

질문 하나에 도구 하나만 부르면 끝나는 게 아니다. 여러 정보가 필요하면 여러 스텝에 걸쳐 도구를 반복 호출해야 한다.

## 실행
```
../.venv/bin/python step4_1_loop.py    # max_steps 안전장치가 있는 기본 루프
../.venv/bin/python step4_2_final.py   # 처리 불가능한 요청에 대한 안내까지 추가
```

## 관찰 포인트
- 콘솔에 찍히는 `[step N] finish_reason=...`을 보면 모델이 몇 번의 왕복 끝에 답을 냈는지 알 수 있다.
- `max_steps`를 일부러 1로 낮춰서 실행해보면 "경고: N step 안에 끝내지 못해 강제 종료" 안전장치가 실제로 동작하는 걸 볼 수 있다.
- `step4_2_final.py`의 "재고가 제일 적은 품목이 뭐야?" 질문은 지원 범위 밖이라, 도구를 잘못 호출하는 대신 안내 메시지로 처리되는지 확인.

---

## English Summary

A single question can need multiple tool calls across multiple round-trips, not just one.

**Run**
```
../.venv/bin/python step4_1_loop.py    # basic loop with a max_steps safety cap
../.venv/bin/python step4_2_final.py   # adds guidance for out-of-scope requests
```

**What to notice**
- The `[step N] finish_reason=...` logs show how many round-trips the model needed to reach an answer.
- Lower `max_steps` to 1 and rerun — you'll see the "forced termination" safety net actually trigger.
- In `step4_2_final.py`, "which item has the lowest stock?" is out of scope — check it gets a guidance message instead of a wrong tool call.
