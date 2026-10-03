"""
Step 12-2: Supervisor 패턴 — LLM이 질문 하나를 보고 "어떤 전문가(들)에게 넘길지" 라우팅한 뒤 결과를 종합한다.

step2_1/step3_3에서 배운 "분류를 여러 선택지 중 하나로 묻지 말고 필드를 쪼개라"는 교훈이 여기서도 재확인된다:
처음엔 needed: list[재고|납기] 하나로 물었더니 모델이 거의 항상 둘 다 골랐다 (README 실전 노트).
needs_inventory/needs_delivery를 독립된 boolean 두 개로 나눠서 "각각 필요한가?"를 따로 묻게 하니
작은 모델도 훨씬 정확하게 판단했다 — "여러 개 중 골라라"보다 "예/아니오를 각각 답해라"가 더 쉬운 문제다.
"""
from openai import OpenAI
from pydantic import BaseModel, Field
from step12_1_specialists import InventorySpecialist, DeliverySpecialist, run_specialist

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

SPECIALISTS = {"재고": InventorySpecialist, "납기": DeliverySpecialist}

ROUTING_SYSTEM = """너는 SCM 문의 라우팅 담당자다. 재고 전문가와 납기 전문가에게 이 질문을 넘길지 각각 독립적으로 판단하라.

- needs_inventory: 창고에 남은 '수량'을 확인해야 답할 수 있으면 true
- needs_delivery: 협력사의 '입고 예정일/지연 여부'를 확인해야 답할 수 있으면 true
- 재고, 납기와 무관한 질문(잡담, 날씨, 음식 등)이면 둘 다 false"""

class RoutingDecision(BaseModel):
    # 두 개를 각각 독립된 bool로 물은 게 핵심 — list[Literal["재고","납기"]] 하나로 물었을 때보다 정확했다.
    needs_inventory: bool = Field(description="재고 수량 확인이 필요하면 true")
    needs_delivery: bool = Field(description="협력사 납기 확인이 필요하면 true")

def route(question: str) -> RoutingDecision:
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": ROUTING_SYSTEM}, {"role": "user", "content": question}],
        temperature=0,
        response_format={"type": "json_schema", "json_schema": {"name": "routing", "schema": RoutingDecision.model_json_schema()}},
    )
    return RoutingDecision.model_validate_json(res.choices[0].message.content)

def supervise(question: str) -> str:
    decision = route(question)
    needed = [n for n, flag in [("재고", decision.needs_inventory), ("납기", decision.needs_delivery)] if flag]
    print(f"  [supervisor] 라우팅 결정: {needed}")

    if not needed:
        return "재고나 납기 관련 질문이 아니라서 답변드릴 수 없습니다."

    reports = []
    for name in needed:
        answer = run_specialist(SPECIALISTS[name], question)
        print(f"  [{name} 전문가] {answer}")
        reports.append(f"[{name} 전문가] {answer}")

    if len(reports) == 1:
        return reports[0].split("] ", 1)[1]

    # 여러 전문가의 답을 하나의 응답으로 종합
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"아래 전문가 보고를 종합해서 한국어로 자연스럽게 하나의 답변으로 만들어라.\n\n{chr(10).join(reports)}"}],
        temperature=0,
    )
    return res.choices[0].message.content

for q in [
    "웨이퍼 재고 얼마나 있어?",
    "웨이퍼 재고랑 B 협력사 납기 상태 둘 다 알려줘",
    "오늘 점심 뭐 먹지?",
]:
    print(f"\n질문: {q}")
    print("최종 답변:", supervise(q))
