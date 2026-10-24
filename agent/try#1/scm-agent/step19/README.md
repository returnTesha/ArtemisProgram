# Step 19 — Observability

에이전트가 여러 단계를 거치면, 답이 이상할 때 "어느 단계에서 틀렸는지" 알아야 고칠 수 있다. 추적(tracing)과 메트릭.

## 실행
```
../.venv/bin/python step19_1_tracing.py   # 함수 호출마다 입력/출력/소요시간을 자동 기록
../.venv/bin/python step19_2_metrics.py   # 여러 요청의 지연시간(p50/p95)과 토큰/비용 집계
```

## 관찰 포인트
- `step19_1_tracing.py`의 `TRACE_LOG` 출력을 보면 `check_stock`(거의 0ms)과 `llm_generate`(수백~수천 ms)의 속도 차이가 확연하다 — 병목이 어디인지 바로 보인다.
- `step19_2_metrics.py`의 p50/p95 지연시간을 평균과 비교해보자 — 평균만 보면 놓치는 "가끔 오래 걸리는 요청"을 percentile이 잡아낸다.
- 로컬 모델은 토큰이 공짜지만, `summary()`의 `예상 비용` 계산식은 실제 유료 API(GPT/Claude 등)를 쓸 때 그대로 적용된다.

---

## English Summary

When an agent has many stages, you need to know which stage went wrong to fix it. Tracing and metrics make that possible.

**Run**
```
../.venv/bin/python step19_1_tracing.py   # auto-log every function's input/output/duration
../.venv/bin/python step19_2_metrics.py   # aggregate latency (p50/p95) and token/cost across requests
```

**What to notice**
- In `step19_1_tracing.py`'s `TRACE_LOG`, notice the stark gap between `check_stock` (near 0ms) and `llm_generate` (hundreds to thousands of ms) — the bottleneck is obvious.
- Compare p50/p95 latency against the average in `step19_2_metrics.py` — averages hide the occasional slow request that percentiles catch.
- Local models make tokens free, but the `estimated cost` formula in `summary()` applies directly once you switch to a paid API (GPT/Claude/etc).
