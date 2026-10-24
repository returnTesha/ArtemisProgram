"""
Step 16-2: 스트리밍 중에도 tool calling(step3~4)을 그대로 할 수 있는가 — 되지만 조립이 필요하다.

일반 텍스트 스트리밍(step16_1)은 delta.content 조각을 이어붙이면 끝이지만,
tool_calls는 "이름 조각", "인자 JSON 문자열 조각"이 여러 chunk에 걸쳐 나눠서 온다.
게다가 한 응답에 tool_call이 여러 개일 수 있으므로, delta.tool_calls[i].index로
"이 조각이 몇 번째 tool_call에 속하는지"를 맞춰가며 각각 따로 이어붙여야 한다 (아래 tool_calls dict).
"""
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

FAKE_STOCK_DB = {"웨이퍼": 1200, "포토레지스트": 340, "불산": 15}

def check_stock(item: str) -> dict:
    qty = FAKE_STOCK_DB.get(item)
    return {"item": item, "quantity": qty} if qty is not None else {"item": item, "found": False}

AVAILABLE_FUNCTIONS = {"check_stock": check_stock}
tools = [{"type": "function", "function": {"name": "check_stock", "description": "재고 조회",
    "parameters": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}}}]

def stream_and_collect(messages):
    stream = client.chat.completions.create(model=MODEL, messages=messages, tools=tools, stream=True)
    content = ""
    tool_calls: dict[int, dict] = {}

    for chunk in stream:
        delta = chunk.choices[0].delta

        if delta.content:
            content += delta.content
            print(delta.content, end="", flush=True)

        if delta.tool_calls:
            print("check tool_calls:", delta.tool_calls)
            for tc_delta in delta.tool_calls:
                # setdefault: 이 index의 tool_call을 처음 보면 빈 슬롯 생성, 이미 있으면 기존 슬롯 재사용.
                slot = tool_calls.setdefault(tc_delta.index, {"id": "", "name": "", "arguments": ""})
                if tc_delta.id:
                    slot["id"] = tc_delta.id
                # name/arguments는 "+="로 이어붙인다 — 한 조각이 아니라 여러 chunk에 걸쳐 나눠져 오기 때문.
                if tc_delta.function.name:
                    slot["name"] += tc_delta.function.name
                if tc_delta.function.arguments:
                    slot["arguments"] += tc_delta.function.arguments

    return content, [tool_calls[i] for i in sorted(tool_calls)]

messages = [{"role": "user", "content": "웨이퍼 재고 알려줘"}]
print("[1차 스트리밍 응답]")
content, tool_calls = stream_and_collect(messages)
print(f"\n조립된 tool_calls: {tool_calls}")

if tool_calls:
    messages.append({"role": "assistant", "content": content or None, "tool_calls": [
        {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in tool_calls
    ]})
    for tc in tool_calls:
        args = json.loads(tc["arguments"])
        result = AVAILABLE_FUNCTIONS[tc["name"]](**args)
        print(f"[도구 실행] {tc['name']}({args}) → {result}")
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result, ensure_ascii=False)})

    print("\n[2차 스트리밍 응답]")
    stream_and_collect(messages)
    print()
