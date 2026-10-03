"""
Step 14-1: step13에서 직접 만든 미니 그래프 엔진을, 실제 LangGraph 라이브러리로 재구현.
(단, 분기는 재고/납기/기타 3개로 줄인 축소판 — step13_2에 있는 규정(policy_node, 임베딩 RAG) 분기는 없다.)

개념은 완전히 같다 — State(TypedDict) + 노드(함수) + add_conditional_edges(분기) —
step13_2_scm_workflow.py와 나란히 놓고 보면 Graph/add_conditional_edge/END가
StateGraph/add_conditional_edges/END로 거의 1:1 대응되는 걸 확인할 수 있다.
차이는 LangGraph가 상태 스키마 검증, 체크포인트(대화 재개), 시각화 같은 걸 대신 해준다는 것 —
직접 만들어본 뒤(step13) 라이브러리로 넘어가야, 그 라이브러리가 실제로 뭘 대신 해주는지 실감할 수 있다.
"""
from typing import Literal, TypedDict
from openai import OpenAI
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

FAKE_STOCK_DB = {"웨이퍼": 1200, "포토레지스트": 340, "불산": 15}
FAKE_DELIVERY_DB = {"B 협력사": {"expected": "2026-09-25", "delayed": True, "delay_days": 3}}

# 그래프 전체가 공유하는 상태. 각 노드는 이 dict의 일부 키만 채워서 반환하고,
# LangGraph가 반환값을 기존 State에 병합해준다 (total=False라 모든 키가 처음부터 있을 필요는 없음).
class State(TypedDict, total=False):
    question: str
    category: str
    tool_result: dict | None
    answer: str

class Category(BaseModel):
    category: Literal["재고", "납기", "기타"]

def classify_node(state: State) -> State:
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"질문을 재고/납기/기타 중 하나로 분류하라.\n\n질문: {state['question']}"}],
        temperature=0,
        response_format={"type": "json_schema", "json_schema": {"name": "cat", "schema": Category.model_json_schema()}},
    )
    return {"category": Category.model_validate_json(res.choices[0].message.content).category}

class ItemExtraction(BaseModel):
    item: str

def inventory_node(state: State) -> State:
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"질문에서 재고를 확인할 품목명만 추출하라.\n\n질문: {state['question']}"}],
        temperature=0,
        response_format={"type": "json_schema", "json_schema": {"name": "item", "schema": ItemExtraction.model_json_schema()}},
    )
    item = ItemExtraction.model_validate_json(res.choices[0].message.content).item
    qty = FAKE_STOCK_DB.get(item)
    return {"tool_result": {"item": item, "quantity": qty} if qty is not None else {"item": item, "found": False}}

class SupplierExtraction(BaseModel):
    supplier: str

def delivery_node(state: State) -> State:
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"질문에서 납기를 확인할 협력사명만 추출하라.\n\n질문: {state['question']}"}],
        temperature=0,
        response_format={"type": "json_schema", "json_schema": {"name": "supplier", "schema": SupplierExtraction.model_json_schema()}},
    )
    supplier = SupplierExtraction.model_validate_json(res.choices[0].message.content).supplier
    match = next((k for k in FAKE_DELIVERY_DB if supplier in k or k in supplier), None)
    return {"tool_result": {"supplier": supplier, **FAKE_DELIVERY_DB[match]} if match else {"supplier": supplier, "found": False}}

def unsupported_node(state: State) -> State:
    return {"tool_result": None}

def respond_node(state: State) -> State:
    if state["tool_result"] is None:
        return {"answer": "재고/납기 관련 질문이 아니라서 답변드릴 수 없습니다."}
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"질문: {state['question']}\n조회 결과: {state['tool_result']}\n\n위 결과를 바탕으로 한국어로 답하라."}],
        temperature=0,
    )
    return {"answer": res.choices[0].message.content}

# classify_node가 정한 category를 보고 다음 노드를 고르는 라우팅 함수.
# step3_3에서 LLM이 tools의 description만 보고 스스로 도구를 골랐던 것과 달리,
# 여기서는 "분류 결과에 따라 어느 노드로 갈지"를 코드가 명시적으로 결정한다 — 그래프 구조의 핵심.
def route_by_category(state: State) -> str:
    return {"재고": "inventory", "납기": "delivery", "기타": "unsupported"}[state["category"]]

builder = StateGraph(State)
builder.add_node("classify", classify_node)
builder.add_node("inventory", inventory_node)
builder.add_node("delivery", delivery_node)
builder.add_node("unsupported", unsupported_node)
builder.add_node("respond", respond_node)

builder.add_edge(START, "classify")
# 조건부 분기: classify 다음에 무조건 한 곳으로 가는 게 아니라, route_by_category의 반환값에 따라 셋 중 하나로 갈라진다.
builder.add_conditional_edges("classify", route_by_category, ["inventory", "delivery", "unsupported"])
for n in ["inventory", "delivery", "unsupported"]:
    builder.add_edge(n, "respond")  # 세 경로 모두 결국 respond로 합류 (다이아몬드 형태)
builder.add_edge("respond", END)

graph = builder.compile()  # 여기서 실제 실행 가능한 그래프로 컴파일된다

for q in ["웨이퍼 재고 얼마나 남았어?", "B 협력사 납기 지연됐어?", "오늘 날씨 어때?"]:
    result = graph.invoke({"question": q})
    print(f"\n질문: {q}")
    print("답변:", result["answer"])
