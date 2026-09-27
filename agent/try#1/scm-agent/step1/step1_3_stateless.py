"""
Step 1-3: LLM API는 상태가 없다(stateless).

서버는 이전 요청을 "기억"하지 않는다. 대화가 이어지는 것처럼 보이려면
매번 지금까지의 대화 전체(messages 배열)를 통째로 다시 보내야 한다.
이 원칙은 이후 모든 step(멀티턴 챗봇, tool calling, RAG)의 기반이 된다.
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

def ask(messages):
    res = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
    print(f"  [토큰] 보낸 것 {res.usage.prompt_tokens} / 받은 것 {res.usage.completion_tokens}")
    return res.choices[0].message.content

print("=== 실험 A: 따로따로 요청 ===")
# 두 번째 호출은 첫 번째 호출과 완전히 무관한 새 요청이다.
# 서버 입장에선 "내 이름은 철수야"라는 문장을 본 적이 없으므로 답을 알 수 없다.
print(ask([{"role": "user", "content": "내 이름은 철수야."}]))
print(ask([{"role": "user", "content": "내 이름이 뭐야?"}]))

print("\n=== 실험 B: 대화 기록을 같이 보냄 ===")
history = [{"role": "user", "content": "내 이름은 철수야."}]
answer = ask(history)
print(answer)

# 모델의 답변(assistant)까지 history에 넣어야, 다음 요청에서 "저 문장에 내가 뭐라고 답했었는지"까지 재구성된다.
history.append({"role": "assistant", "content": answer})
history.append({"role": "user", "content": "내 이름이 뭐야?"})
# 이번엔 history 전체(3개 메시지)가 매번 다시 전송되기 때문에 이름을 "기억"하는 것처럼 보인다.
# 실제로는 기억이 아니라, 매번 전체 대화를 처음부터 다시 읽는 것뿐이다 (그래서 토큰 사용량이 턴이 갈수록 늘어난다).
print(ask(history))
