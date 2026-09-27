"""
Step 3-4: step3 시리즈의 완성본 — 도구 여러 개 + 실제 실행 + system 프롬프트로 행동 지침 명시.

주목할 실험 포인트: system 프롬프트(40번 줄)엔 "재고/납기는 도구로 확인하라"고만 적혀 있고,
"날씨 같은 무관한 질문은 거부해라"는 말이 전혀 없다. 그런데도 "오늘 날씨 어때?"를 물으면 모델이 거부한다.
이건 system 프롬프트 문장 때문이 아니라, tools 파라미터 자체가 있다는 사실만으로도
모델이 "내 역할은 이 도구들 범위 안"이라고 스스로 좁혀서 판단하는 경향(tool-anchoring) 때문이다.
(tools 없이, system 프롬프트도 없이 순수 채팅으로 물으면 오히려 "날씨 앱 써보세요" 같은 일반 상식 답을 준다 — 직접 비교해볼 것)
"""
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

FAKE_STOCK_DB = {"웨이퍼": 1200, "포토레지스트": 340, "불산": 15}
FAKE_DELIVERY_DB = {
    "B 협력사": {"expected": "2026-09-25", "delayed": True, "delay_days": 3},
    "C사": {"expected": "2026-09-22", "delayed": False, "delay_days": 0},
}

def check_stock(item: str) -> dict:
    qty = FAKE_STOCK_DB.get(item)
    if qty is None:
        return {"item": item, "found": False}
    return {"item": item, "found": True, "quantity": qty, "unit": "kg"}

def check_delivery(supplier: str) -> dict:
    info = FAKE_DELIVERY_DB.get(supplier)
    if info is None:
        return {"supplier": supplier, "found": False}
    return {"supplier": supplier, "found": True, **info}

AVAILABLE_FUNCTIONS = {"check_stock": check_stock, "check_delivery": check_delivery}

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "품목명으로 현재 창고 재고 수량을 조회한다.",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string", "description": "조회할 품목명. 문의에서 언급된 정확한 명칭 사용"}},
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_delivery",
            "description": "협력사명으로 입고 예정일과 지연 여부를 조회한다. 문의에 협력사 이름이 하나라도 언급되면 반드시 이 함수를 호출한다.",
            "parameters": {
                "type": "object",
                "properties": {"supplier": {"type": "string", "description": "문의에 언급된 협력사명을 그대로 사용 (예: 'B 협력사', 'C사')"}},
                "required": ["supplier"],
            },
        },
    },
]

def run_agent(question: str):
    messages = [
        {"role": "system", "content": "너는 SCM 도우미다. 재고나 납기를 물으면 반드시 도구를 사용해서 실제 값을 확인한 뒤 답한다."},
        {"role": "user", "content": question},
    ]
    res = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    msg = res.choices[0].message
    messages.append(msg)

    if not msg.tool_calls:
        return msg.content

    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        result = AVAILABLE_FUNCTIONS[tc.function.name](**args)
        print(f"  [도구 실행] {tc.function.name}({args}) → {result}")
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False)})

    res2 = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    return res2.choices[0].message.content

for q in ["불산 재고 얼마나 남았어?", "B 협력사 납기 언제야?", "오늘 날씨 어때?"]:
    print(f"\n질문: {q}")
    print("답변:", run_agent(q))