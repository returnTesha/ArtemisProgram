# Step 1 — LLM 호출 기초

LLM API는 상태가 없다(stateless). 매 요청마다 대화 기록 전체를 다시 보내야 "기억"하는 것처럼 보인다.

## 실행
```
../.venv/bin/python step1_1_call.py         # 가장 기본적인 단일 호출
../.venv/bin/python step1_2_temperature.py  # temperature가 답변 다양성에 미치는 영향
../.venv/bin/python step1_3_stateless.py    # 대화 기록을 안 보내면 왜 "기억"을 못 하는지
../.venv/bin/python step1_4_chatbot.py      # 토큰 한도에 따라 오래된 대화를 잘라내는 실전 챗봇
```

## 관찰 포인트
- `step1_3_stateless.py`의 실험 A/B를 비교하면 "기억"이 모델의 능력이 아니라 우리가 보낸 메시지 배열의 결과라는 게 보인다.
- `step1_2_temperature.py`에서 temperature=0은 매번 거의 같은 답, 1.5는 매번 다른 답 — 이 차이를 눈으로 확인해보자.
- `step1_4_chatbot.py`는 토큰이 쌓이면 오래된 질문-답변 쌍을 지운다. 지워진 이전 대화를 물어보면 어떻게 되는지 직접 실험해보면 좋다.

---

## English Summary

LLM APIs are stateless. To "remember" a conversation, you must resend the entire history every time.

**Run**
```
../.venv/bin/python step1_1_call.py         # a single basic call
../.venv/bin/python step1_2_temperature.py  # how temperature affects variety
../.venv/bin/python step1_3_stateless.py    # why the model "forgets" without full history
../.venv/bin/python step1_4_chatbot.py      # a chatbot that trims old turns when tokens run high
```

**What to notice**
- Comparing experiments A/B in `step1_3_stateless.py` shows "memory" is just what's in the message array, not a model capability.
- `temperature=0` gives near-identical answers each time; `1.5` gives different answers every run.
- `step1_4_chatbot.py` deletes old Q&A pairs once the token budget is exceeded — try asking about a deleted turn.
