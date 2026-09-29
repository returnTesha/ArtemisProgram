"""
Step 9-1: RAG 검색(step5~8의 임베딩+유사도)을 "그냥 또 하나의 tool"로 등록한다.

step3~4에서 check_stock/check_delivery 같은 실시간 DB 조회 함수를 tools에 등록했던 것과
완전히 같은 방식으로, search_policy(임베딩 코사인 유사도 검색 함수)도 그대로 tools에 등록한다.
즉 LLM 입장에서는 "실시간 DB를 조회하는 도구"와 "문서를 검색하는 도구"가 구조적으로 다르지 않다 —
description만 보고 둘 중 뭘 쓸지(혹은 둘 다 쓸지, 아무것도 안 쓸지) 스스로 판단한다 (step3_3에서 배운 원칙 그대로).
여기서는 search_policy 하나만 있는 가장 단순한 형태이고, step9_2에서 실시간 도구와 합쳐진다.
"""
import json, sys, os
import numpy as np
from openai import OpenAI
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "step6"))
from docs import SCM_DOCS

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"

def embed(text: str) -> np.ndarray:
    return np.array(client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 문서 전체를 미리(요청 들어오기 전에) 한 번만 임베딩해서 벡터로 캐싱해둔다.
# 매 검색마다 문서를 다시 임베딩하면 느리고 낭비이므로, "질문만" 그때그때 임베딩해서 이 벡터들과 비교한다.
DOC_VECTORS = [embed(d) for d in SCM_DOCS]

def search_policy(query: str) -> dict:
    # 코사인 유사도 상위 3개만 반환 (step5_3에서 본 "TOP_K로 추리기"와 같은 발상)
    query_vec = embed(query)
    scored = sorted(((cosine_similarity(query_vec, v), d) for v, d in zip(DOC_VECTORS, SCM_DOCS)), reverse=True)
    return {"results": [d for _, d in scored[:3]]}

AVAILABLE_FUNCTIONS = {"search_policy": search_policy}

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "사내 SCM 규정(발주 절차, 재고 기준, 협력사 평가, 검수 규정 등)을 검색한다. 규정/절차/기준을 묻는 질문이면 반드시 호출한다.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "검색할 질문이나 키워드"}},
                "required": ["query"],
            },
        },
    }
]

def run_agent(question: str):
    messages = [
        {"role": "system", "content": "너는 SCM 도우미다. 규정이나 절차를 묻는 질문이면 search_policy 도구로 반드시 검색한 뒤, 검색 결과에 근거해서만 답한다."},
        {"role": "user", "content": question},
    ]
    res = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    msg = res.choices[0].message
    messages.append(msg)

    if not msg.tool_calls:
        return msg.content

    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        result = AVAILABLE_FUNCTIONS[tc.function.name](**args)
        print(f"  [도구 실행] search_policy({args}) → {len(result['results'])}건")
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False)})

    res2 = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    return res2.choices[0].message.content

for q in ["신규 협력사 등록 절차 알려줘", "오늘 점심 뭐 먹지?"]:
    print(f"\n질문: {q}")
    print("답변:", run_agent(q))
