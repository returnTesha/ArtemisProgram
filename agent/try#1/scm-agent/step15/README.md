# Step 15 — 자율 루프 & Reflection

답을 한 번에 확정하지 않고 스스로 비평 후 개선하는 self-reflection, 목표만 주고 언제 끝낼지도 스스로 정하는 자율 루프.

## 실행
```
../.venv/bin/python step15_1_reflection.py         # 초안 → 비평 → 개선 반복
../.venv/bin/python step15_2_autonomous_loop.py    # finish_investigation을 스스로 호출해 종료
```

## 관찰 포인트
- `step15_1_reflection.py`가 `[비평] 문제 없음 → 종료`로 끝날 수도 있다 — reflection은 항상 답을 고치는 게 아니라, "고칠 게 없다"는 판단도 정상적인 결과다.
- `step15_2_autonomous_loop.py`의 로그를 보면 모델이 스스로 `finish_investigation`을 호출하지 않고 그냥 텍스트로 끝내버리는 경우가 있다 — 이때 "강제 종료" 안전장치가 실제로 작동하는지 확인해보자.
- 이 강제 종료 케이스가 바로 "모델이 시킨 대로 완벽하게 행동하지 않을 수 있다"는 걸 보여주는 실제 사례다. max_steps 같은 안전장치가 왜 항상 필요한지 체감할 수 있다.

---

## English Summary

Self-reflection: don't commit to an answer immediately — critique it, then revise. Autonomous loop: given only a goal, the agent decides for itself when to stop.

**Run**
```
../.venv/bin/python step15_1_reflection.py         # draft → critique → revise
../.venv/bin/python step15_2_autonomous_loop.py    # the agent calls finish_investigation itself
```

**What to notice**
- `step15_1_reflection.py` can end with `[비평] 문제 없음 → 종료` (no issues found) — reflection doesn't always mean revising; "nothing to fix" is a valid outcome too.
- In `step15_2_autonomous_loop.py`, the model sometimes ends with plain text instead of calling `finish_investigation` — check that the forced-termination safety net kicks in.
- That forced-termination path is a real example of "the model won't always follow instructions perfectly," and why a `max_steps` safety net is always necessary.
