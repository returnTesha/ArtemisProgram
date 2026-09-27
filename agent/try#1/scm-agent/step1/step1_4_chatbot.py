"""
Step 1-4: 실제로 손으로 치면서 쓰는 대화형 챗봇.

step1_3에서 배운 "전체 history를 매번 다시 보낸다"는 원칙을 그대로 실전에 적용한다.
다만 history가 무한정 커지면 토큰(=비용/속도/컨텍스트 윈도우)이 감당 안 되므로,
토큰 한도를 넘으면 가장 오래된 대화부터 잘라내는 슬라이딩 윈도우를 더했다.
(자동으로 여러 턴을 흘려보내며 이 truncation을 직접 관찰하고 싶으면 step1_4_chatbot_demo.py 참고)
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"
MAX_TOKENS = 500  # 이 한도를 넘기면 가장 오래된 대화 1턴을 지운다 (34번 줄)

SYSTEM = {"role": "system", "content": "너는 SCM 업무를 돕는 사내 도우미다. 한국어로 간결하게 답한다."}
history = [SYSTEM]

print("SCM 도우미입니다. 종료: quit / 초기화: /reset")

while True:
    user_input = input("\n나: ").strip()

    if user_input == "quit":
        break
    if user_input == "/reset":
        history = [SYSTEM]
        print("대화 기록을 초기화했어요.")
        continue
    if not user_input:
        continue

    history.append({"role": "user", "content": user_input})

    res = client.chat.completions.create(model=MODEL, messages=history, temperature=0.3)
    answer = res.choices[0].message.content
    history.append({"role": "assistant", "content": answer})

    print(f"도우미: {answer}")
    print(f"  [보낸 토큰 {res.usage.prompt_tokens} / 대화 {len(history) // 2}턴]")

    # 토큰이 한도를 넘으면 가장 오래된 질문-답변 한 쌍 삭제 (system은 유지)
    if res.usage.prompt_tokens > MAX_TOKENS and len(history) > 3:
        del history[1:3]
        print("  [오래된 대화 1턴 삭제]")