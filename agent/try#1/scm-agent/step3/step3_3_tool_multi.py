"""
Step 3-3: 도구가 여러 개일 때, "어떤 도구를 쓸지 / 아예 안 쓸지"는 전적으로 LLM의 판단이다.

우리 코드는 세 질문 모두 똑같은 코드(tools=tools)로 처리한다 — if-else 분기 로직이 전혀 없다.
LLM은 오직 tools 리스트에 적힌 name/description/parameters(메타데이터)만 보고
- 질문이 어떤 도구의 description과 의미적으로 맞는지 판단해서 그 도구를 선택하거나,
- 어느 도구의 description과도 안 맞으면 tool_calls를 아예 만들지 않고 자연어로 직접 답한다.
그래서 실무에서 "에이전트가 엉뚱한 도구를 고른다"는 문제가 생기면, 가장 먼저 의심할 곳은
코드 로직이 아니라 description 문구다 — description이 라우팅 정확도를 좌우한다.
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "품목명으로 현재 창고 재고 수량을 조회한다.",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string", "description": "조회할 품목명"}},
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_delivery",
            "description": "협력사명으로 입고 예정일과 지연 여부를 조회한다.",
            "parameters": {
                "type": "object",
                "properties": {"supplier": {"type": "string", "description": "조회할 협력사명"}},
                "required": ["supplier"],
            },
        },
    },
]

questions = [
    "불산 재고 얼마나 남았어?",      # check_stock의 description과 매칭 → 그 도구 선택
    "B 협력사 납기 언제야?",         # check_delivery의 description과 매칭 → 그 도구 선택
    "오늘 날씨 어때?",  # 둘 다 아닌 경우 → 도구를 호출 안 하고 자연어로 직접 답해야 정상
]

for q in questions:
    res = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": q}],
        tools=tools,
    )
    msg = res.choices[0].message
    print(f"\n질문: {q}")
    if msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  → 도구 선택: {tc.function.name}({tc.function.arguments})")
    else:
        print(f"  → 도구 없이 직접 답: {msg.content}")