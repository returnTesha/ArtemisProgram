"""
Step 11-1: ReAct(Reasoning + Acting) — step4의 tool calling 루프에 "생각을 말로 출력"하는 걸 더한 패턴.

step4/9의 루프도 내부적으론 매 스텝 판단을 하지만, 그 판단 과정이 안 보였다 (tool_calls만 툭 나옴).
ReAct는 SYSTEM 프롬프트로 "Thought: ..."를 먼저 말하게 강제해서, 모델이 왜 그 도구를 선택했는지
추론 과정 자체를 로그로 남긴다 — 정확도를 높이기보다는 "디버깅 가능성"을 높이는 패턴에 가깝다.
"""
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

FAKE_STOCK_DB = {"웨이퍼": 1200, "포토레지스트": 340, "불산": 15}
FAKE_DELIVERY_DB = {"B 협력사": {"expected": "2026-09-25", "delayed": True, "delay_days": 3}}

def check_stock(item: str) -> dict:
    qty = FAKE_STOCK_DB.get(item)
    return {"item": item, "found": True, "quantity": qty} if qty is not None else {"item": item, "found": False}

def check_delivery(supplier: str) -> dict:
    info = FAKE_DELIVERY_DB.get(supplier)
    if info is None:
        match = next((k for k in FAKE_DELIVERY_DB if supplier in k or k in supplier), None)
        info = FAKE_DELIVERY_DB.get(match) if match else None
    return {"supplier": supplier, "found": True, **info} if info else {"supplier": supplier, "found": False}

AVAILABLE_FUNCTIONS = {"check_stock": check_stock, "check_delivery": check_delivery}
tools = [
    {"type": "function", "function": {"name": "check_stock", "description": "재고 조회",
        "parameters": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}}},
    {"type": "function", "function": {"name": "check_delivery", "description": "납기 조회",
        "parameters": {"type": "object", "properties": {"supplier": {"type": "string"}}, "required": ["supplier"]}}},
]

# ReAct = Reasoning + Acting을 매 스텝마다 "말로" 반복시키는 패턴.
# tool_calls만 쓰면 모델이 왜 그 도구를 골랐는지가 안 보이는데,
# 매번 "생각 → 행동"을 명시적으로 출력하게 하면 추론 과정 자체를 확인/디버깅할 수 있다.
SYSTEM = """너는 SCM 도우미다. 매 단계마다 다음 형식을 반드시 지켜라:

Thought: (지금까지 알아낸 것과 다음에 무엇을 확인해야 하는지 한국어로 추론)

그 다음, 확인이 더 필요하면 도구를 호출하고, 충분하면 도구 호출 없이 최종 답을 말해라."""

def run_react(question: str, max_steps: int = 4):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]

    for step in range(1, max_steps + 1):
        res = client.chat.completions.create(model=MODEL, messages=messages, tools=tools, temperature=0)
        msg = res.choices[0].message
        messages.append(msg)

        print(f"\n[Step {step}]")
        if msg.content:
            print(msg.content)  # Thought 부분

        if not msg.tool_calls:
            return msg.content

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = AVAILABLE_FUNCTIONS[tc.function.name](**args)
            print(f"Action: {tc.function.name}({args})")
            print(f"Observation: {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False)})

    return "최대 스텝을 초과했습니다."

question = "웨이퍼 재고 확인하고, B 협력사 납기도 지연됐는지 같이 알려줘"
print(f"질문: {question}")
answer = run_react(question)
print(f"\n최종 답변: {answer}")
