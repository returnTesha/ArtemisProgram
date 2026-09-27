"""
Step 3-2: tool calling의 전체 왕복(round-trip)을 완성한다.

step3_1에서 멈췄던 "LLM의 제안"에서 이어서, 실제 실행 → 결과를 대화에 반영 → 최종 답변까지 6단계를 완성:
  1~2) 사용자 질문 → 1차 호출 → LLM이 tool_calls 제안
  3~4) 우리 코드가 실제 함수 실행 → 결과를 role="tool" 메시지로 messages에 추가
  5~6) 그 결과를 포함한 전체 history로 2차 호출 → LLM이 결과를 보고 자연어 최종 답변 생성
핵심 원칙(step1_3의 stateless)이 여기서도 그대로 적용된다: 2차 호출 때 모델이 "기억"하는 게 아니라,
1차 요청+제안+실행결과가 전부 담긴 messages 배열을 처음부터 다시 통째로 읽는 것뿐이다.
"""
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

FAKE_STOCK_DB = {"웨이퍼": 1200, "포토레지스트": 340, "불산": 15}

def check_stock(item: str) -> dict:
    qty = FAKE_STOCK_DB.get(item)
    if qty is None:
        return {"item": item, "found": False}
    return {"item": item, "found": True, "quantity": qty, "unit": "kg"}

# 함수 이름(문자열) → 실제 Python 함수를 연결하는 표
AVAILABLE_FUNCTIONS = {"check_stock": check_stock}

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "품목명으로 현재 창고 재고 수량을 조회한다.",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string", "description": "조회할 품목명"}},
                "required": ["item"],
            },
        },
    }
]

messages = [{"role": "user", "content": "불산 재고 얼마나 남았어?"}]

# --- 1, 2번: 첫 호출 ---
res = client.chat.completions.create(model="qwen2.5:7b", messages=messages, tools=tools)
msg = res.choices[0].message
# msg는 dict가 아니라 pydantic 객체(ChatCompletionMessage)지만, 그대로 append해도 된다.
# SDK가 다음 요청을 만들 때 이 객체를 알아서 올바른 JSON({"role":"assistant","tool_calls":[...]})으로 직렬화해준다.
messages.append(msg)  # LLM의 응답(tool_calls 포함)도 history에 쌓는다

print("check:", res)

print("[1차 호출] finish_reason:", res.choices[0].finish_reason)

# --- 3, 4번: 도구 실행하고 결과를 messages에 추가 ---
if msg.tool_calls:
    for tc in msg.tool_calls:
        func_name = tc.function.name
        func_args = json.loads(tc.function.arguments)  # 문자열 → 딕셔너리
        print(f"[실행] {func_name}({func_args})")

        func = AVAILABLE_FUNCTIONS[func_name]
        result = func(**func_args)  # 실제 Python 함수 호출!
        print(f"[결과] {result}")

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,  # 어떤 요청에 대한 결과인지 짝을 맞춤 (tool_call이 여러 개일 수 있어서 필요, step3_3 참고)
            "content": json.dumps(result, ensure_ascii=False),
        })

    # --- 5, 6번: 결과를 본 LLM이 최종 답변 생성 ---
    # 이 시점의 messages = [user 질문, assistant(tool_calls 제안), tool(실행 결과)] 전부를 다시 통째로 보낸다.
    # 모델이 "방금 일을 기억"하는 게 아니라, 이 전체 텍스트를 처음부터 다시 읽고 답을 만드는 것 (stateless 원칙 재적용).
    res2 = client.chat.completions.create(model="qwen2.5:7b", messages=messages, tools=tools)
    print("\n[2차 호출] finish_reason:", res2.choices[0].finish_reason)
    print("최종 답변:", res2.choices[0].message.content)