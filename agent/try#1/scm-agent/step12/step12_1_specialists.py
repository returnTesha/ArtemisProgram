"""
Step 12-1: 멀티 에이전트의 재료 — 도구를 분야별로 쪼갠 "전문가" 에이전트 두 개.

여기선 아직 라우팅이 없다. 각 전문가를 그냥 직접 호출해서 개별로 잘 동작하는지만 확인한다.
"누가 어떤 질문에 어떤 전문가를 부를지" 결정하는 로직(supervisor)은 step12_2에서 추가된다.
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

# 전문가(specialist) 에이전트 = 자기 분야 도구만 가진, 좁고 깊은 에이전트.
# 하나의 에이전트에 도구를 다 몰아넣는 대신 역할을 쪼개면, 각 프롬프트가 짧고 명확해져서
# 작은 모델(7b)도 헷갈리지 않고, 나중에 전문가를 독립적으로 교체/개선하기 쉬워진다.

class InventorySpecialist:
    system = "너는 재고 전문가다. check_stock 도구로만 재고를 확인해서 한국어로 간결히 답한다."
    tools = [{"type": "function", "function": {"name": "check_stock", "description": "재고 조회",
        "parameters": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}}}]
    functions = {"check_stock": check_stock}

class DeliverySpecialist:
    system = "너는 납기 전문가다. check_delivery 도구로만 납기를 확인해서 한국어로 간결히 답한다."
    tools = [{"type": "function", "function": {"name": "check_delivery", "description": "납기 조회",
        "parameters": {"type": "object", "properties": {"supplier": {"type": "string"}}, "required": ["supplier"]}}}]
    functions = {"check_delivery": check_delivery}

def run_specialist(specialist, question: str) -> str:
    messages = [{"role": "system", "content": specialist.system}, {"role": "user", "content": question}]
    res = client.chat.completions.create(model=MODEL, messages=messages, tools=specialist.tools, temperature=0)
    msg = res.choices[0].message
    if not msg.tool_calls:
        return msg.content
    messages.append(msg)
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        result = specialist.functions[tc.function.name](**args)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False)})
    return client.chat.completions.create(model=MODEL, messages=messages).choices[0].message.content

if __name__ == "__main__":
    print("재고 전문가:", run_specialist(InventorySpecialist, "웨이퍼 재고 얼마나 남았어?"))
    print("납기 전문가:", run_specialist(DeliverySpecialist, "B 협력사 납기 지연됐어?"))
