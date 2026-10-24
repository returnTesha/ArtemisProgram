# Step 17 — 가드레일 & 견고성

입력 검증(공격 차단), 출력 필터링(정보 유출 방지), 재시도(일시적 장애 대응) — 프로덕션에서 안 죽는 에이전트 만들기.

## 실행
```
../.venv/bin/python step17_1_input_guard.py    # 규칙 기반 + LLM 기반 이중 입력 가드
../.venv/bin/python step17_2_output_guard.py   # 시스템 프롬프트 유출 차단, PII 마스킹
../.venv/bin/python step17_3_retry.py          # 지수 백오프로 타임아웃 재시도
```

## 관찰 포인트
- `step17_1_input_guard.py`의 "이전 지시 무시" 같은 직접 명령은 규칙 기반으로 즉시 차단되지만, "DAN" 역할극은 LLM 판단 단계까지 가서야 걸린다 — 두 겹 방어가 왜 필요한지 확인.
- `step17_2_output_guard.py`의 전화번호가 `[전화번호 마스킹됨]`으로 바뀌는지 확인해보자.
- `step17_3_retry.py`의 마지막 실험에서 `timeout=0.01`로 일부러 타임아웃을 내보면, 실제 `APITimeoutError`가 발생하고 `call_with_retry`가 1초 → 2초 → 4초 백오프로 재시도하다가 끝내 `RuntimeError`로 실패하는 걸 볼 수 있다 — 이게 실전에서 재시도 로직이 다뤄야 하는 진짜 예외다.

---

## English Summary

Input validation (block attacks), output filtering (prevent leaks), and retries (survive transient failures) — building an agent that doesn't die in production.

**Run**
```
../.venv/bin/python step17_1_input_guard.py    # two-layer guard: rule-based + LLM-based
../.venv/bin/python step17_2_output_guard.py   # blocks system-prompt leaks, masks PII
../.venv/bin/python step17_3_retry.py          # exponential backoff on timeouts
```

**What to notice**
- A direct command like "ignore previous instructions" gets blocked instantly by the rule-based layer, while a "DAN" roleplay jailbreak only gets caught at the LLM-judgment stage — see why both layers are needed.
- Confirm a phone number gets masked as `[전화번호 마스킹됨]` in `step17_2_output_guard.py`.
- The last experiment in `step17_3_retry.py` deliberately sets `timeout=0.01` — you'll see a real `APITimeoutError` on every attempt, while `call_with_retry` backs off 1s → 2s → 4s and finally raises `RuntimeError` — the exact exception retry logic needs to handle in production.
