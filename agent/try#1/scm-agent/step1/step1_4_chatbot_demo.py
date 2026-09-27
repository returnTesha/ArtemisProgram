"""
Step 1-4 (비대화형 데모): step1_4_chatbot.py의 truncation 동작을 손으로 안 치고 자동으로 확인.

input()으로 사람이 매턴 입력해야 하는 원본과 달리, 5턴짜리 대화를 미리 코드에 박아두고
자동으로 흘려보내면서 "몇 번째 턴에서 오래된 대화가 삭제되는지"를 로그로 보여준다.
결론: 삭제되는 건 항상 가장 오래된(먼저 나눈) 대화이지, 최근 대화가 아니다.
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"
MAX_TOKENS = 500  # 일부러 낮게 잡아서 몇 턴 안 가서 삭제가 일어나도록 함

SYSTEM = {"role": "system", "content": "너는 SCM 업무를 돕는 사내 도우미다. 한국어로 간결하게 답한다."}
history = [SYSTEM]

# input() 대신 미리 정해둔 5턴짜리 대화로 자동 진행
questions = [
    "B 협력사 웨이퍼 납기가 3일 늦어진대요. 왜 이런 상황이 생겼는지 원인 좀 정리해줘",
    "그럼 대체 협력사로 돌릴 수 있는지도 같이 알려줘",
    "재고로 며칠이나 버틸 수 있는지 계산 방법도 설명해줘",
    "지금까지 얘기한 내용 중에 제일 처음에 말한 '납기 지연 원인'이 뭐였는지 기억해?",
    "마지막으로 전체 상황 한 줄 요약해줘",
]

for turn, user_input in enumerate(questions, start=1):
    print(f"\n[{turn}턴] 나: {user_input}")

    history.append({"role": "user", "content": user_input})

    res = client.chat.completions.create(model=MODEL, messages=history, temperature=0.3)
    answer = res.choices[0].message.content
    history.append({"role": "assistant", "content": answer})

    print(f"도우미: {answer}")
    print(f"  [보낸 토큰 {res.usage.prompt_tokens} / history 길이 {len(history)} / 대화 {len(history) // 2}턴]")

    # 토큰이 한도를 넘으면 가장 오래된 질문-답변 한 쌍 삭제 (system은 유지)
    if res.usage.prompt_tokens > MAX_TOKENS and len(history) > 3:
        dropped = history[1:3]
        del history[1:3]
        print(f"  [오래된 대화 1턴(가장 먼저 나눈 질문/답변) 삭제됨 → 삭제된 질문: \"{dropped[0]['content'][:30]}...\"]")
        print(f"  [삭제 후 history 길이 {len(history)}]")

print("\n--- 종료: 최종 history에 남아있는 대화 ---")
for m in history:
    if m["role"] != "system":
        print(f"  {m['role']}: {m['content'][:40]}")
