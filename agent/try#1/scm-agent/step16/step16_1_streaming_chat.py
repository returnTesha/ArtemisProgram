"""
Step 16-1: stream=False vs stream=True — 같은 요청인데 "체감 속도"가 완전히 다르다.

stream=False면 답변이 다 만들어질 때까지 기다렸다가 한 번에 받는다.
stream=True면 토큰이 생성되는 즉시 조각(chunk)으로 받을 수 있다 — 챗봇 UI의 "타이핑 중" 효과의 정체다.
"전체 완료까지 걸리는 시간"은 두 방식이 비슷하다 — 서버가 실제로 계산하는 양은 똑같기 때문.
차이는 "첫 토큰이 나오기까지"의 시간이고, 이게 사용자가 느끼는 응답성을 좌우한다.
"""
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

print("=== stream=False (완성될 때까지 대기) ===")
start = time.time()
res = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "SCM이 뭔지 세 문장으로 설명해줘"}])
print(f"[{time.time() - start:.1f}초 후 전체 출력]")
print(res.choices[0].message.content)

print("\n=== stream=True (토큰 단위로 즉시 출력) ===")
start = time.time()
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "SCM이 뭔지 세 문장으로 설명해줘"}],
    stream=True,
)
first_token_time = None
full_text = ""
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        if first_token_time is None:
            first_token_time = time.time() - start
        print(delta, end="", flush=True)
        full_text += delta
print(f"\n\n[첫 토큰까지 {first_token_time:.2f}초, 전체 완료까지 {time.time() - start:.1f}초]")
print("→ 전체 완료 시간은 비슷하지만, 사용자는 첫 토큰이 나오는 순간부터 읽기 시작할 수 있다. 체감 응답 속도가 완전히 다르다.")
