"""
Step 4-1: 도구 하나로 안 끝나는 질문 — 여러 스텝에 걸친 tool calling 루프 + 안전장치.

step3까지는 "1차 호출(도구 제안) → 실행 → 2차 호출(최종 답)"로 정확히 2번만 호출했다.
하지만 복합 질문(예: "재고 제일 적은 거 찾고, 그 다음 협력사 납기도 확인해줘")은
한 번의 실행으로 안 끝날 수 있어서, tool_calls가 나오는 동안 계속 반복하는 for 루프가 필요하다.
max_steps는 "모델이 끝없이 도구만 호출하며 답을 안 내는" 상황을 막는 안전장치.

관찰 포인트: 아래 질문은 사실 지금 도구로는 "완전히는" 답할 수 없다 —
check_stock은 품목명을 알아야 조회 가능한 도구이지, "전체 중 제일 적은 걸 찾아줘" 같은
목록 조회 기능이 없다. 그래서 모델이 가짜 품목명을 지어내 호출했다가 실패하고,
답할 수 있었을 나머지 절반(B 협력사 납기)까지 통째로 포기하는 걸 볼 수 있다.
→ 이 실패가 바로 step4_2에서 "지원하지 않는 요청" 가드레일을 추가하는 이유가 된다.
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
    return {"item": item, "found": qty is not None, "quantity": qty, "unit": "kg"} if qty else {"item": item, "found": False}

def check_delivery(supplier: str) -> dict:
    info = FAKE_DELIVERY_DB.get(supplier)
    return {"supplier": supplier, "found": True, **info} if info else {"supplier": supplier, "found": False}

AVAILABLE_FUNCTIONS = {"check_stock": check_stock, "check_delivery": check_delivery}

tools = [
    {"type": "function", "function": {
        "name": "check_stock", "description": "품목명으로 현재 창고 재고 수량을 조회한다.",
        "parameters": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}}},
    {"type": "function", "function": {
        "name": "check_delivery", "description": "협력사명으로 입고 예정일과 지연 여부를 조회한다. 협력사 이름이 언급되면 반드시 호출한다.",
        "parameters": {"type": "object", "properties": {"supplier": {"type": "string"}}, "required": ["supplier"]}}},
]

def run_agent(question: str, max_steps: int = 4):
    messages = [
        {"role": "system", "content": "너는 SCM 도우미다. 재고나 납기를 물으면 반드시 도구를 사용해서 실제 값을 확인한 뒤 답한다. 여러 정보가 필요하면 필요한 도구를 순서대로 모두 사용한다. 도구로 답할 수 없는 요청이면 무엇을 확인할 수 없는지 한국어로 명확히 설명한다."},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        res = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        msg = res.choices[0].message
        messages.append(msg)
        print(f"  [step {step}] finish_reason={res.choices[0].finish_reason}")

        # tool_calls가 없다는 건 "더 이상 도구가 필요 없다, 이게 최종 답이다"라는 뜻 → 루프 종료
        if not msg.tool_calls:
            return msg.content

        # tool_calls가 있으면 루프를 안 끝내고 계속 돈다 — 한 번의 실행으로 안 끝나는 질문을 처리하는 핵심
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = AVAILABLE_FUNCTIONS[tc.function.name](**args)
            print(f"    [도구 실행] {tc.function.name}({args}) → {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False)})

    # 여기 도달 = max_steps를 다 썼는데도 못 끝냄(무한 루프 방지) → 강제로 마지막 답변 요구
    print(f"  [경고] {max_steps} step 안에 끝내지 못해 강제 종료")
    messages.append({"role": "user", "content": "지금까지 확인한 정보만으로 한국어로 간단히 답해줘. 확인 못 한 부분은 확인 못 했다고 말해줘."})
    res = client.chat.completions.create(model=MODEL, messages=messages)
    return res.choices[0].message.content


for q in [
    "재고가 제일 적은 품목이 뭔지 확인하고, 그 품목을 만드는 데 필요한 협력사 중에 B 협력사 납기가 지연됐는지도 알려줘",
]:
    print(f"\n질문: {q}")
    print("답변:", run_agent(q))