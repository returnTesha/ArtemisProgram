"""
Step 3-1: Tool calling(함수 호출)의 가장 기본적인 형태 — LLM은 "제안"만 하고 "실행"은 안 한다.

tools 리스트로 함수 명세(이름/설명/파라미터 스키마)를 LLM에게 알려주면,
LLM은 사용자 질문을 보고 어떤 함수를, 어떤 인자로 부르고 싶은지를 구조화된 JSON(tool_calls)으로 돌려준다.
중요: 이 시점에 check_stock()이 실제로 실행되는 게 아니다. 그건 여전히 우리 코드가 해야 할 일 —
이 파일은 딱 "LLM이 뭘 하고 싶어하는지 확인"까지만 하고, 실제 실행+결과 반영은 step3_2에서 완성된다.
"""
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# 가짜 재고 DB
FAKE_STOCK_DB = {
    "웨이퍼": 1200,
    "포토레지스트": 340,
    "불산": 15,
}

def check_stock(item: str) -> dict:
    """실제로는 여기서 DB를 조회한다. 지금은 가짜 데이터로 흉내만 낸다."""
    qty = FAKE_STOCK_DB.get(item)
    if qty is None:
        return {"item": item, "found": False}
    return {"item": item, "found": True, "quantity": qty, "unit": "kg"}

# LLM에게 알려줄 "함수 명세서". LLM은 description 텍스트만 보고 이 함수를 쓸지 말지, 뭘 넘길지 판단한다.
tools = [
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "품목명으로 현재 창고 재고 수량을 조회한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "조회할 품목명"}
                },
                "required": ["item"],
            },
        },
    }
]

res = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "불산 재고 얼마나 남았어?"}],
    tools=tools,
)
print("res:", res)

msg = res.choices[0].message
# finish_reason="tool_calls"면 도구 호출을 "제안"한 것 — 아직 실행된 게 아니다.
print("finish_reason:", res.choices[0].finish_reason)
print("content:", msg.content)  # 도구를 호출하는 경우 보통 비어 있다 (할 말을 tool_calls로 대신 전달)
print("tool_calls:", msg.tool_calls)  # 함수 이름 + 인자(JSON 문자열)만 담긴 "제안서"