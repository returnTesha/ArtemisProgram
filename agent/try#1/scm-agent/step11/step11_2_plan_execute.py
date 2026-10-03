"""
Step 11-2: Plan-and-Execute — 계획을 전부 세운 뒤 순서대로 실행하는 패턴 (ReAct의 대안).

make_plan()이 질문 하나를 하위 작업 목록으로 쪼개고(계획 단계는 도구 없이 순수 텍스트 분해),
execute_step()이 각 작업을 "새로운" 대화로 독립 실행한다(step1_3 stateless 원칙 — 이전 스텝의
대화 기록을 안 물려주고 매번 새 messages로 시작). 마지막에 모든 실행 결과를 모아 한 번에 종합 답변을 만든다.
"""
import json
from openai import OpenAI
from pydantic import BaseModel

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

# ReAct: 한 스텝씩 보고 즉흥적으로 다음 행동을 정한다 (반응형).
# Plan-and-Execute: 먼저 할 일 전체를 계획으로 쪼갠 뒤, 순서대로 실행한다 (계획형).
# 계획형은 전체 그림을 미리 확정하므로 스텝 수를 예측 가능하고, 계획 단계만 검토/수정하기 쉽다.
# 대신 실행 중 예상 못한 상황에는 ReAct보다 유연하게 대응하지 못한다.

class Plan(BaseModel):
    steps: list[str]  # 실행할 하위 작업 목록, 자연어 문장

def make_plan(question: str) -> Plan:
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"다음 요청을 처리하기 위한 하위 작업 목록을 순서대로 나눠라. 각 작업은 재고 조회 또는 납기 조회 중 하나여야 한다.\n\n요청: {question}"}],
        temperature=0,
        response_format={"type": "json_schema", "json_schema": {"name": "plan", "schema": Plan.model_json_schema()}},
    )
    return Plan.model_validate_json(res.choices[0].message.content)

def execute_step(step_text: str) -> dict:
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            # 실전에서 겪은 버그(README 참고): "B 협력사"가 도구 인자로 전달되는 과정에서 "B 協力公司"처럼
            # 한자가 섞여 깨진 적이 있다. 계획 단계 → 실행 단계로 텍스트가 한 번 더 LLM을 거치면서
            # 고유명사가 "번역/변형"될 위험이 생기므로, 이렇게 명시적으로 금지해줘야 한다.
            {"role": "system", "content": "도구를 호출할 때 지시문에 나온 고유명사(품목명, 협력사명)를 절대 번역하거나 변형하지 말고 원문 그대로 인자로 사용하라."},
            {"role": "user", "content": step_text},
        ],
        tools=tools,
        temperature=0,
    )
    msg = res.choices[0].message
    if not msg.tool_calls:
        return {"step": step_text, "result": msg.content}
    tc = msg.tool_calls[0]
    args = json.loads(tc.function.arguments)
    result = AVAILABLE_FUNCTIONS[tc.function.name](**args)
    return {"step": step_text, "tool": tc.function.name, "args": args, "result": result}

def run_plan_execute(question: str):
    plan = make_plan(question)
    print("계획:")
    for i, s in enumerate(plan.steps, 1):
        print(f"  {i}. {s}")

    observations = []
    for step_text in plan.steps:
        obs = execute_step(step_text)
        print(f"\n실행: {obs}")
        observations.append(obs)

    summary_prompt = f"원래 질문: {question}\n\n실행 결과: {json.dumps(observations, ensure_ascii=False)}\n\n위 결과를 종합해서 한국어로 답하라."
    res = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": summary_prompt}], temperature=0)
    return res.choices[0].message.content

question = "웨이퍼 재고 확인하고, B 협력사 납기도 지연됐는지 같이 알려줘"
print(f"질문: {question}\n")
print("\n최종 답변:", run_plan_execute(question))
