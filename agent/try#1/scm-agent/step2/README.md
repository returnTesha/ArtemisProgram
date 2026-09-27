# Step 2 — 구조화된 출력

"JSON으로만 답해"라는 프롬프트만으로는 파싱이 깨진다. JSON Schema로 강제하고, Pydantic으로 검증해야 안전하다.

## 실행
```
../.venv/bin/python step2_1_prompt_json.py   # 프롬프트만으로 JSON 요청 → 종종 파싱 실패
../.venv/bin/python step2_2_schema_json.py   # JSON Schema로 강제 → 파싱 실패가 사라짐
../.venv/bin/python step2_3_pydantic.py      # Pydantic으로 검증까지 추가
../.venv/bin/python step2_4_reason_order.py  # 필드 순서(reason이 먼저냐 나중이냐)가 정확도에 미치는 영향
../.venv/bin/python step2_5_eval_classify.py  # classifier.py를 여러 케이스로 정확도 측정
```

## 관찰 포인트
- `step2_1_prompt_json.py`를 여러 번 돌려보면 가끔 파싱이 실패한다 — 이게 스키마 강제가 필요한 이유다.
- **"파싱 성공"과 "값이 맞음"은 다르다.** `step2_1`에서 `json.loads()`가 통과했다고 안심하면 안 된다. 모델이 `{"category": "발주|재고|납기|기타", ...}`처럼 문법적으로는 완벽한 JSON을 뱉어도, `category` 값 자체는 어떤 카테고리에도 속하지 않는 쓰레기일 수 있다. `step2_3_pydantic.py`가 `Literal[...]`로 이 값을 실제로 검증해서 걸러내는 걸 보자 — "JSON이다"와 "우리가 원하는 스키마를 만족한다"는 완전히 다른 검증 단계다.
- `step2_4_reason_order.py`에서 `ReasonFirst`가 `ReasonLast`보다 정확도가 높게 나오는 경우를 확인해보자 (모델이 답을 먼저 정하고 나중에 근거를 짜맞추는 걸 막는 효과).
  - **왜 이런 일이 생기나:** pydantic `BaseModel`에 필드를 선언한 순서가 그대로 `model_json_schema()`의 `properties` 순서가 되고, `response_format: json_schema`로 구조화 출력을 걸면 모델은 그 스키마의 key 순서 그대로 토큰을 생성한다. LLM은 왼쪽에서 오른쪽으로, 이미 뱉은 토큰을 보면서 다음 토큰을 만드는 구조(autoregressive)이기 때문에 — `reason`이 `category`/`urgency`보다 먼저 오면 모델이 "생각을 다 쓴 다음 결론을 내림"이 되고, `reason`이 나중에 오면 "결론부터 정해놓고 근거는 사후 짜맞추기"가 된다. **즉 내가 pydantic 필드를 어떤 순서로 선언하느냐가 곧 모델의 사고 순서를 설계하는 것**이다 — schema는 단순 검증 도구가 아니라 모델의 추론 흐름을 제어하는 레버다.
- `step2_5_eval_classify.py`의 정확도가 100%가 아니어도 정상이다 — 작은 로컬 모델의 한계를 그대로 보여준다.

## 이 스텝에서 챙겨가야 할 것
자유 텍스트로 응답받는 건 데모에서나 통하고, 실제 에이전트에서는 이 JSON 출력이 다음 단계(라우팅, 툴 호출, DB 저장)의 **입력**이 된다. 형식이 한 번이라도 깨지면 그 뒤 파이프라인 전체가 죽는다. 그래서 에이전트 엔지니어는 "모델이 알아서 잘 답하겠지"가 아니라 "형식을 구조적으로 강제하고, 값까지 검증하고, 그 검증 실패조차 예상된 시나리오로 다뤄야 한다"는 태도로 접근한다. 여기서 배운 스키마 설계(필드 순서 포함)는 이후 스텝의 tool-calling, 멀티 에이전트 라우팅에서 그대로 재사용된다.

---

## English Summary

Asking for "JSON only" in a prompt isn't enough — parsing breaks. Enforce it with a JSON Schema and validate with Pydantic.

**Run**
```
../.venv/bin/python step2_1_prompt_json.py   # prompt-only JSON request → parsing often fails
../.venv/bin/python step2_2_schema_json.py   # JSON Schema enforcement → failures disappear
../.venv/bin/python step2_3_pydantic.py      # add Pydantic validation on top
../.venv/bin/python step2_4_reason_order.py  # field order (reason first vs last) affects accuracy
../.venv/bin/python step2_5_eval_classify.py  # measure classifier.py accuracy across test cases
```

**What to notice**
- Run `step2_1_prompt_json.py` a few times — parsing occasionally fails, which is exactly why schema enforcement matters.
- **"Parses" and "is correct" are not the same thing.** Don't relax just because `json.loads()` succeeded in `step2_1`. The model can emit syntactically perfect JSON like `{"category": "발주|재고|납기|기타", ...}` where the `category` value doesn't belong to any real category at all. Watch `step2_3_pydantic.py` use `Literal[...]` to actually catch that value — "it's valid JSON" and "it satisfies the schema we want" are two separate checks.
- In `step2_4_reason_order.py`, `ReasonFirst` often beats `ReasonLast` — forcing the model to reason before committing to an answer instead of rationalizing after.
  - **Why this happens:** the order you declare fields in a pydantic `BaseModel` becomes the `properties` order in `model_json_schema()`, and once you enforce that schema via `response_format: json_schema`, the model generates JSON keys in exactly that order. LLMs are autoregressive — each token is generated conditioned on the tokens already written — so putting `reason` before `category`/`urgency` forces "think, then conclude," while putting it last produces "decide first, rationalize after the fact." **The field order you choose is literally designing the model's order of reasoning** — a schema isn't just a validation tool, it's a lever on the model's inference flow.
- `step2_5_eval_classify.py` not hitting 100% is expected — it's an honest look at a small local model's limits.

## What to take away from this step
Free-text responses only work in demos. In a real agent, this JSON output becomes the **input** to the next stage — routing, tool calls, database writes. If the format breaks even once, the whole downstream pipeline dies. So an agent engineer's default stance isn't "the model will probably answer nicely" — it's "structurally enforce the format, validate the values, and treat validation failure itself as an expected scenario to handle." The schema design skills here (including field ordering) carry straight into tool-calling and multi-agent routing in later steps.
