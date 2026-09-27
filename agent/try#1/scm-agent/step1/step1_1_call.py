"""
Step 1-1: 가장 단순한 LLM 호출 한 번.

로컬에 떠 있는 Ollama(포트 11434)를 OpenAI SDK로 부른다.
Ollama가 OpenAI 호환 API를 제공하기 때문에, base_url만 로컬로 바꾸면
평소 쓰던 openai 패키지 코드를 그대로 재사용할 수 있다.
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# api_key="ollama"는 실제 인증 키가 아니라 더미 값이다.
# 로컬 서버라 인증이 필요 없지만, OpenAI SDK가 client 생성 시 빈 문자열을 거부하므로 아무 값이나 채워 넣은 것.

res = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        # system: 모델의 "역할/규칙"을 정하는 지시문. 대화 내용이 아니라 행동 지침.
        {"role": "system", "content": "너는 한 문장으로만 답한다."},
        # user: 실제 사용자 질문.
        {"role": "user", "content": "REST API가 뭐야?"},
    ],
    temperature=0.7,  # 값이 높을수록 답이 매번 달라짐 (step1_2에서 직접 비교)
)

print("Answer:", res.choices[0].message.content)
# finish_reason: 모델이 "왜" 응답을 멈췄는지. stop=자연스럽게 끝, length=max_tokens 초과, tool_calls=도구 호출 요청(step3~).
print("Finish Reason:", res.choices[0].finish_reason)
# 토큰 수 = 과금/컨텍스트윈도우 사용량의 기준. prompt=보낸 것, completion=받은 것.
print("Tokens:", res.usage.prompt_tokens, "+", res.usage.completion_tokens)
