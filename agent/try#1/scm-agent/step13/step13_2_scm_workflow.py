"""
Step 13-2: step13_1의 Graph 엔진으로 지금까지의 SCM 에이전트를 "노드 + 분기"로 재구성한다.

step12의 supervisor(라우팅 결정 → 전문가 호출 → 종합)와 하려는 일은 비슷하지만, 표현 방식이 다르다:
supervisor는 if/for로 흐름을 직접 코드에 적었고, 여기선 classify_node의 결과에 따라
route_by_category()가 다음 노드(inventory/delivery/policy/unsupported)를 결정하는 "그래프"로 표현한다.
흐름이 복잡해질수록(분기가 여러 겹, 되돌아가는 경로 등) 그래프 형태가 코드보다 추적/수정하기 쉬워진다는 게 핵심.
각 노드는 독립적으로 자기 할 일(분류, 추출, 조회)만 하고 state 딕셔너리로 다음 노드에 정보를 넘긴다.
"""
import sys, os
from typing import Literal
import numpy as np
from openai import OpenAI
from pydantic import BaseModel
from step13_1_graph_engine import Graph, END
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "step6"))
from docs import SCM_DOCS

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"

FAKE_STOCK_DB = {"웨이퍼": 1200, "포토레지스트": 340, "불산": 15}
FAKE_DELIVERY_DB = {"B 협력사": {"expected": "2026-09-25", "delayed": True, "delay_days": 3}}

def embed(text: str) -> np.ndarray:
    return np.array(client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

DOC_VECTORS = [embed(d) for d in SCM_DOCS]

# --- 노드들: 각자 state를 받아서 state를 갱신해 돌려주는 순수 함수 ---

class Category(BaseModel):
    category: Literal["재고", "납기", "규정", "기타"]

def classify_node(state: dict) -> dict:
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"질문을 재고/납기/규정/기타 중 하나로 분류하라.\n\n질문: {state['question']}"}],
        temperature=0,
        response_format={"type": "json_schema", "json_schema": {"name": "cat", "schema": Category.model_json_schema()}},
    )
    state["category"] = Category.model_validate_json(res.choices[0].message.content).category
    return state

class ItemExtraction(BaseModel):
    item: str

def inventory_node(state: dict) -> dict:
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"질문에서 재고를 확인할 품목명만 추출하라.\n\n질문: {state['question']}"}],
        temperature=0,
        response_format={"type": "json_schema", "json_schema": {"name": "item", "schema": ItemExtraction.model_json_schema()}},
    )
    item = ItemExtraction.model_validate_json(res.choices[0].message.content).item
    qty = FAKE_STOCK_DB.get(item)
    state["tool_result"] = {"item": item, "quantity": qty} if qty is not None else {"item": item, "found": False}
    return state

class SupplierExtraction(BaseModel):
    supplier: str

def delivery_node(state: dict) -> dict:
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"질문에서 납기를 확인할 협력사명만 추출하라.\n\n질문: {state['question']}"}],
        temperature=0,
        response_format={"type": "json_schema", "json_schema": {"name": "supplier", "schema": SupplierExtraction.model_json_schema()}},
    )
    supplier = SupplierExtraction.model_validate_json(res.choices[0].message.content).supplier
    match = next((k for k in FAKE_DELIVERY_DB if supplier in k or k in supplier), None)
    state["tool_result"] = {"supplier": supplier, **FAKE_DELIVERY_DB[match]} if match else {"supplier": supplier, "found": False}
    return state

def policy_node(state: dict) -> dict:
    query_vec = embed(state["question"])
    scored = sorted(((cosine_similarity(query_vec, v), d) for v, d in zip(DOC_VECTORS, SCM_DOCS)), reverse=True)
    state["tool_result"] = {"policy_docs": [d for _, d in scored[:2]]}
    return state

def unsupported_node(state: dict) -> dict:
    state["tool_result"] = None
    return state

def respond_node(state: dict) -> dict:
    if state["tool_result"] is None:
        state["answer"] = "SCM(재고/납기/규정) 관련 질문이 아니라서 답변드릴 수 없습니다."
        return state
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"질문: {state['question']}\n조회 결과: {state['tool_result']}\n\n위 결과를 바탕으로 한국어로 답하라."}],
        temperature=0,
    )
    state["answer"] = res.choices[0].message.content
    return state

def route_by_category(state: dict) -> str:
    return {"재고": "inventory", "납기": "delivery", "규정": "policy", "기타": "unsupported"}[state["category"]]

# --- 그래프 조립 ---
g = Graph()
g.add_node("classify", classify_node)
g.add_node("inventory", inventory_node)
g.add_node("delivery", delivery_node)
g.add_node("policy", policy_node)
g.add_node("unsupported", unsupported_node)
g.add_node("respond", respond_node)

g.set_entry("classify")
g.add_conditional_edge("classify", route_by_category)
for n in ["inventory", "delivery", "policy", "unsupported"]:
    g.add_edge(n, "respond")
g.add_edge("respond", END)

for q in [
    "웨이퍼 재고 얼마나 남았어?",
    "B 협력사 납기 지연됐어?",
    "신규 협력사 등록하려면 뭘 통과해야 해?",
    "오늘 날씨 어때?",
]:
    result = g.run({"question": q})
    print(f"\n질문: {q}")
    print("경로:", " → ".join(result["_trace"]))
    print("답변:", result["answer"])
