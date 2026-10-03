"""
Step 14-2: step3~4에서 직접 짰던 tool-calling 루프를, LangChain의 prebuilt ReAct agent로 대체.

`create_agent` 한 줄이 "tool_calls 파싱 → 실행 → 결과를 tool 메시지로 다시 넣기 → 재호출" 루프
(step4_2_final.py의 run_agent 함수 전체)를 대신해준다. 내부적으로 하는 일은 완전히 동일하다 —
그 루프를 손으로 직접 짜본 적이 있어야, 이 한 줄이 실제로 뭘 감추고 있는지 실감할 수 있다.

실전 노트(step14 README 참고): 원래 `langgraph.prebuilt.create_react_agent`를 쓰려 했으나
실행 시 deprecation 경고가 떠서 최신 API인 `langchain.agents.create_agent`로 바꿨다.
LangChain/LangGraph 생태계는 API가 빠르게 바뀌므로, 설치된 버전 기준으로 공식 문서를 확인하는 습관이 중요하다.
"""
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

llm = ChatOpenAI(base_url="http://localhost:11434/v1", api_key="ollama", model="qwen2.5:7b", temperature=0)

FAKE_STOCK_DB = {"웨이퍼": 1200, "포토레지스트": 340, "불산": 15}
FAKE_DELIVERY_DB = {"B 협력사": {"expected": "2026-09-25", "delayed": True, "delay_days": 3}}

# @tool 데코레이터가 함수의 시그니처(타입 힌트)와 docstring을 읽어서
# step3의 tools=[{"type":"function", "function":{"name":..., "description":..., "parameters":...}}]와
# 동일한 스키마를 자동으로 만들어준다 — docstring이 곧 LLM에게 보여줄 description이 된다.
@tool
def check_stock(item: str) -> dict:
    """품목명으로 현재 창고 재고 수량을 조회한다."""
    qty = FAKE_STOCK_DB.get(item)
    return {"item": item, "quantity": qty} if qty is not None else {"item": item, "found": False}

@tool
def check_delivery(supplier: str) -> dict:
    """협력사명으로 입고 예정일과 지연 여부를 조회한다."""
    match = next((k for k in FAKE_DELIVERY_DB if supplier in k or k in supplier), None)
    return {"supplier": supplier, **FAKE_DELIVERY_DB[match]} if match else {"supplier": supplier, "found": False}

agent = create_agent(
    model=llm,
    tools=[check_stock, check_delivery],
    system_prompt="너는 SCM 도우미다. 반드시 한국어로만 답하고, 재고나 납기를 물으면 도구로 확인한 뒤 답한다.",
)

for q in ["웨이퍼랑 불산 재고 둘 다 알려줘", "B 협력사 납기 지연됐으면 며칠이나 지연됐는지도 알려줘"]:
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})
    print(f"\n질문: {q}")
    for m in result["messages"]:
        role = getattr(m, "type", m.__class__.__name__)
        if role == "tool":
            print(f"  [도구 결과] {m.content}")
        elif role == "ai" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                print(f"  [도구 호출] {tc['name']}({tc['args']})")
    print("답변:", result["messages"][-1].content)
