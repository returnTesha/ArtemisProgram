import time
from openai import OpenAI, APITimeoutError, APIConnectionError

"""
Step 17-3: 지수 백오프(exponential backoff) 재시도 — 일시적 장애에서 죽지 않는 에이전트.

프로덕션 에이전트가 죽는 가장 흔한 이유는 모델 서버가 순간적으로 느려지거나(타임아웃),
네트워크가 잠깐 끊기는 것이다. 재시도 없이 바로 예외를 던지면 사용자 입장에선 "그냥 에러".
지수 백오프: 실패할 때마다 대기 시간을 2배씩 늘려서 재시도한다 — 서버가 과부하 상태일 때
똑같은 요청을 곧바로 다시 퍼부으면(고정 간격 재시도) 오히려 상황을 악화시키기 때문이다.
"""
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

def call_with_retry(messages: list[dict], max_retries: int = 3, base_delay: float = 1.0, timeout: float = 30.0):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(model=MODEL, messages=messages, timeout=timeout)
        except (APITimeoutError, APIConnectionError) as e:
            last_error = e
            wait = base_delay * (2 ** (attempt - 1))
            print(f"  [시도 {attempt}/{max_retries} 실패: {type(e).__name__}] {wait:.1f}초 후 재시도")
            time.sleep(wait)
    raise RuntimeError(f"{max_retries}번 재시도했지만 실패했습니다: {last_error}")

print("=== 정상 케이스 (타임아웃 넉넉함) ===")
res = call_with_retry([{"role": "user", "content": "SCM을 한국어 한 단어로 요약해줘"}], timeout=30.0)
print("답변:", res.choices[0].message.content)

print("\n=== 매 시도가 타임아웃나도록 만든 경우 (아주 짧은 timeout=0.01) ===")
# 모델 응답이 timeout(0.01초)을 넘기므로 APITimeoutError가 매 시도마다 진짜로 발생한다.
# call_with_retry가 1초 → 2초 → 4초 간격으로 재시도하다가, 끝내 실패하면 RuntimeError를 던진다.
try:
    call_with_retry([{"role": "user", "content": "안녕"}], max_retries=3, base_delay=1.0, timeout=0.01)
except RuntimeError as e:
    print(f"  최종 실패: {e}")
