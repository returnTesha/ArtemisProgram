"""
Step 4-2: step4_1의 실패("재고 제일 적은 품목" 같은 지원 불가 요청)를 system 프롬프트로 고친다.

핵심 변경: SYSTEM에 "[지원하지 않는 요청]" 섹션을 명시해서, 그런 질문엔 아예 도구를 호출하지 말고
바로 안내 메시지로 답하도록 지시했다. 실행해보면 "재고가 제일 적은 품목이 뭐야?"가
가짜 인자로 도구를 잘못 호출하지 않고 곧바로 안내 답변으로 처리되는 걸 확인할 수 있다 (의도대로 성공).

주의: 같은 실행에서도 "B 협력사 납기 지연됐어?"처럼 단순한 질문이 가끔 이상하게 새는 경우가 있는데
(tool_calls 대신 "<tool_call>" 같은 원본 템플릿 텍스트가 content로 새어나오거나, 답변 중간에
갑자기 무관한 언어로 바뀌는 등) — 이건 이 system 프롬프트 때문이 아니라, temperature가
고정 안 된 상태에서 작은 로컬 모델(qwen2.5:7b)이 보이는 비결정적 불안정성이다. 같은 질문을
temperature=0 없이 여러 번 돌려보면 매번 다른 결과가 나오는 걸 확인할 수 있다.
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
    # 부분 일치(fuzzy matching): 모델이 "B 협력사"를 "B"처럼 줄여서 넘기는 경우가 있어서
    # 정확히 일치하는 키가 없으면 한쪽이 다른 쪽 문자열에 포함되는지로 한 번 더 찾는다.
    info = FAKE_DELIVERY_DB.get(supplier)
    if info is None:
        match = next((k for k in FAKE_DELIVERY_DB if supplier in k or k in supplier), None)
        info = FAKE_DELIVERY_DB.get(match) if match else None
    if info is None:
        return {"supplier": supplier, "found": False}
    return {"supplier": supplier, "found": True, **info}

AVAILABLE_FUNCTIONS = {"check_stock": check_stock, "check_delivery": check_delivery}

tools = [
    {"type": "function", "function": {
        "name": "check_stock", "description": "품목명으로 현재 창고 재고 수량을 조회한다.",
        "parameters": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}}},
    {"type": "function", "function": {
        "name": "check_delivery", "description": "협력사명으로 입고 예정일과 지연 여부를 조회한다. 협력사 이름이 언급되면 반드시 호출한다.",
        "parameters": {"type": "object", "properties": {"supplier": {"type": "string"}}, "required": ["supplier"]}}},
]

SYSTEM = """너는 SCM 도우미다. 반드시 한국어로만 답한다.

재고나 납기를 물으면 반드시 도구를 사용해서 실제 값을 확인한 뒤 답한다.
여러 정보가 필요하면 필요한 도구를 순서대로 모두 사용한다.

[지원하지 않는 요청]
- 재고가 가장 적은/많은 품목 찾기, 전체 품목 목록 조회, 품목 간 비교는 지원하지 않는다.
- 이런 요청을 받으면 도구를 호출하지 말고, 어떤 품목명을 알려주면 개별 조회가 가능한지 안내한다."""
# ↑ 이 섹션이 step4_1의 문제(존재하지 않는 "목록 조회" 능력을 흉내내려다 가짜 인자로 실패)를 막는 핵심.
# "할 수 없는 걸 억지로 하려 하지 말고, 할 수 없다고 솔직히 말해라"를 명시적으로 가르친 것.

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

    print(f"  [경고] {max_steps} step 안에 끝내지 못해 강제 종료")
    messages.append({"role": "user", "content": "지금까지 확인한 정보만으로 한국어로 간단히 답해줘."})
    res = client.chat.completions.create(model=MODEL, messages=messages)
    return res.choices[0].message.content

for q in [
    "B 협력사 납기 지연됐어?",
    "웨이퍼랑 불산 재고 둘 다 알려줘",
    "재고가 제일 적은 품목이 뭐야?",
]:
    print(f"\n질문: {q}")
    print("답변:", run_agent(q))