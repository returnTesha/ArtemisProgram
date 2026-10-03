import json, sys, os
import numpy as np
from openai import OpenAI
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "step6"))
from docs import SCM_DOCS

"""
Step 15-2: 자율 루프(autonomous loop) — 목표만 주어지고, 언제 그만할지도 에이전트 스스로 정한다.

지금까지(step4 등)의 멀티스텝 루프는 "더 이상 tool_calls가 없으면 그게 최종 답"이라고 해석했다.
여기서는 다르다 — "조사를 끝내겠다"는 판단 자체가 명시적으로 finish_investigation이라는
하나의 tool call이 되도록 설계했다. 그래서 이 루프에서 tool_calls 없이 그냥 텍스트로 끝나는 건
"정상 종료"가 아니라 "모델이 지시(반드시 finish_investigation을 호출하라)를 못 지킨 상황"으로 취급하고
강제 종료 처리한다 (run_autonomous 안의 "도구 호출 없이 텍스트만 응답" 분기). 모델이 시킨 대로 완벽하게 행동하지 않을 수 있다는 걸 보여주는 실제 사례이자,
max_steps 같은 안전장치가 왜 항상 필요한지 체감할 수 있는 지점이다.
"""
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"

FAKE_STOCK_DB = {"웨이퍼": 1200, "포토레지스트": 340, "불산": 15}
FAKE_DELIVERY_DB = {"B 협력사": {"expected": "2026-09-25", "delayed": True, "delay_days": 3}}

def check_stock(item: str) -> dict:
    qty = FAKE_STOCK_DB.get(item)
    return {"item": item, "quantity": qty} if qty is not None else {"item": item, "found": False}

def check_delivery(supplier: str) -> dict:
    match = next((k for k in FAKE_DELIVERY_DB if supplier in k or k in supplier), None)
    return {"supplier": supplier, **FAKE_DELIVERY_DB[match]} if match else {"supplier": supplier, "found": False}

def embed(text: str) -> np.ndarray:
    return np.array(client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

DOC_VECTORS = [embed(d) for d in SCM_DOCS]

def search_policy(query: str) -> dict:
    query_vec = embed(query)
    scored = sorted(((cosine_similarity(query_vec, v), d) for v, d in zip(DOC_VECTORS, SCM_DOCS)), reverse=True)
    return {"results": [d for _, d in scored[:2]]}

# "끝났다"는 판단 자체를 일반 함수 호출처럼 스키마화한 것 — __finished__ 플래그를
# run_autonomous 루프의 result.get("__finished__") 검사가 감지해서 정상 종료 신호로 쓴다. 실제 DB 조회는 없는 가짜 도구.
def finish_investigation(report: str) -> dict:
    return {"__finished__": True, "report": report}

AVAILABLE_FUNCTIONS = {
    "check_stock": check_stock,
    "check_delivery": check_delivery,
    "search_policy": search_policy,
    "finish_investigation": finish_investigation,
}

tools = [
    {"type": "function", "function": {"name": "check_stock", "description": "품목 재고 수량 조회",
        "parameters": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}}},
    {"type": "function", "function": {"name": "check_delivery", "description": "협력사 납기 조회",
        "parameters": {"type": "object", "properties": {"supplier": {"type": "string"}}, "required": ["supplier"]}}},
    {"type": "function", "function": {"name": "search_policy", "description": "사내 규정/기준 검색",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "finish_investigation",
        "description": "조사가 충분히 끝나서 최종 보고서를 제출할 준비가 됐을 때 호출한다.",
        "parameters": {"type": "object", "properties": {"report": {"type": "string", "description": "지금까지 확인한 내용을 종합한 한국어 최종 보고서"}}, "required": ["report"]}}},
]

SYSTEM = """너는 SCM 리스크를 스스로 조사하는 자율 에이전트다.
주어진 목표를 달성하기 위해 필요한 도구를 스스로 판단해서 순서대로 호출하라.
충분히 확인했다고 판단되면 반드시 finish_investigation을 호출해서 조사를 종료하라."""

def run_autonomous(goal: str, max_steps: int = 6):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"목표: {goal}"}]

    for step in range(1, max_steps + 1):
        res = client.chat.completions.create(model=MODEL, messages=messages, tools=tools, temperature=0)
        msg = res.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            print(f"  [step {step}] 도구 호출 없이 텍스트만 응답 → 강제 종료")
            return msg.content

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = AVAILABLE_FUNCTIONS[tc.function.name](**args)
            print(f"  [step {step}] {tc.function.name}({args}) → {result}")

            if result.get("__finished__"):
                return result["report"]

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False)})

    return "(안전장치) 최대 스텝에 도달해 조사를 강제 종료했습니다."

goal = "웨이퍼 생산 라인이 멈추지 않도록 지금 위험 요소가 있는지 확인하고 보고해줘"
print(f"목표: {goal}\n")
print("최종 보고서:\n", run_autonomous(goal))
