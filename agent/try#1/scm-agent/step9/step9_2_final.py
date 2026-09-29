"""
Step 9-2: 실시간 DB 조회 도구(check_stock/check_delivery)와 정적 문서 검색 도구(search_policy)를 한 에이전트에 합친다.

관건은 "언제 어떤 도구를 쓰는가"를 LLM이 헷갈리지 않게 만드는 것 — 그래서 SYSTEM 프롬프트(SYSTEM 변수, 63번 줄 근처)에서
"실시간 데이터=check_stock/check_delivery, 규정/기준=search_policy"라고 명시적으로 구분해준다.
이 구분이 없으면 "기준을 넘었는지"처럼 실시간 값과 규정값을 둘 다 비교해야 하는 질문에서
모델이 도구 하나만 쓰고 섣불리 답하거나 엉뚱한 도구를 고르기 쉽다.
step4의 다단계 루프(max_steps) 구조를 재사용해서, 두 도구를 순서대로 여러 번 호출하는 것도 지원한다.
다만 한 가지 다르다: step4는 max_steps를 다 쓰면 "강제로 마지막 답변을 요구하는" 추가 LLM 호출을 하지만,
여기(run_agent)는 그런 호출 없이 고정 문자열 "확인하는 데 시간이 너무 오래 걸려서 중단했습니다."를 그대로 반환한다.
"""
import json, sys, os
import numpy as np
from openai import OpenAI
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "step6"))
from docs import SCM_DOCS

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"

# --- step3/4에서 쓰던 실시간 데이터 조회 도구 ---
FAKE_STOCK_DB = {"웨이퍼": 1200, "포토레지스트": 340, "불산": 15}
FAKE_DELIVERY_DB = {
    "B 협력사": {"expected": "2026-09-25", "delayed": True, "delay_days": 3},
    "C사": {"expected": "2026-09-22", "delayed": False, "delay_days": 0},
}

def check_stock(item: str) -> dict:
    qty = FAKE_STOCK_DB.get(item)
    return {"item": item, "found": True, "quantity": qty, "unit": "kg"} if qty is not None else {"item": item, "found": False}

def check_delivery(supplier: str) -> dict:
    info = FAKE_DELIVERY_DB.get(supplier)
    return {"supplier": supplier, "found": True, **info} if info else {"supplier": supplier, "found": False}

# --- step6/9에서 쓰던 정적 지식(사내 규정) 검색 도구 ---
def embed(text: str) -> np.ndarray:
    return np.array(client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

DOC_VECTORS = [embed(d) for d in SCM_DOCS]

def search_policy(query: str) -> dict:
    query_vec = embed(query)
    scored = sorted(((cosine_similarity(query_vec, v), d) for v, d in zip(DOC_VECTORS, SCM_DOCS)), reverse=True)
    return {"results": [d for _, d in scored[:3]]}

AVAILABLE_FUNCTIONS = {"check_stock": check_stock, "check_delivery": check_delivery, "search_policy": search_policy}

tools = [
    {"type": "function", "function": {
        "name": "check_stock", "description": "품목명으로 현재 창고 재고 '수량'을 조회한다 (실시간 데이터).",
        "parameters": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}}},
    {"type": "function", "function": {
        "name": "check_delivery", "description": "협력사명으로 입고 예정일과 지연 여부를 조회한다 (실시간 데이터).",
        "parameters": {"type": "object", "properties": {"supplier": {"type": "string"}}, "required": ["supplier"]}}},
    {"type": "function", "function": {
        "name": "search_policy", "description": "발주 절차, 안전 재고 '기준', 협력사 평가 방식 같은 사내 규정을 검색한다 (정적 문서 지식).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]

SYSTEM = """너는 SCM 도우미다. 반드시 한국어로만 답한다.

- '지금 수량이 얼마인지', '언제 도착하는지' 같은 실시간 데이터는 check_stock / check_delivery로 확인한다.
- '기준이 뭔지', '절차가 어떻게 되는지' 같은 규정/정책은 search_policy로 확인한다.
- 둘 다 필요하면 (예: 현재 수량이 기준을 넘었는지) 두 도구를 순서대로 모두 사용해서 직접 비교 판단한다."""

def run_agent(question: str, max_steps: int = 4):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]

    for step in range(1, max_steps + 1):
        res = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        msg = res.choices[0].message
        messages.append(msg)
        print(f"  [step {step}] finish_reason={res.choices[0].finish_reason}")

        if not msg.tool_calls:
            return msg.content

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = AVAILABLE_FUNCTIONS[tc.function.name](**args)
            print(f"    [도구 실행] {tc.function.name}({args}) → {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False)})

    return "확인하는 데 시간이 너무 오래 걸려서 중단했습니다."

for q in [
    "웨이퍼 재고가 안전 수량 기준을 넘었는지 확인해줘",
    "B 협력사 납기 지연됐으면 대체 소싱 검토해야 하는 규정에 해당돼?",
]:
    print(f"\n질문: {q}")
    print("답변:", run_agent(q))
