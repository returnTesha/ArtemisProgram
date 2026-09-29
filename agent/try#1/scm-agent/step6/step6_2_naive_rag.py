"""
Step 6-2: RAG 세 단계(Retrieve → Augment → Generate)를 프레임워크/벡터DB 없이 순수 코드로 구현.

step5에서 배운 "임베딩 + 코사인 유사도로 유사 문장 찾기"를 그대로 재사용하되,
찾아낸 문서를 실제로 프롬프트에 끼워 넣어(augment) 최종 답변까지 생성하는 게 step5와의 차이다.
프롬프트에 "규정에 없으면 모른다고 답하라"는 지시를 넣은 게 핵심 안전장치 —
관련 없는 문서를 억지로 끌어와 그럴듯하게 답하는(hallucination) 걸 막는다.
"""
import numpy as np
from openai import OpenAI
from docs import SCM_DOCS

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
CHAT_MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"

def embed(text: str):
    res = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(res.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 실제로는 매번 임베딩하면 느리니 미리 다 벡터로 만들어 메모리에 올려둔다.
print("문서 임베딩 중...")
DOC_VECTORS = [embed(d) for d in SCM_DOCS]

def retrieve(query: str, top_k: int = 3) -> list[str]:
    query_vec = embed(query)
    scored = [(cosine_similarity(query_vec, v), d) for v, d in zip(DOC_VECTORS, SCM_DOCS)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for score, d in scored[:top_k]]

def answer(question: str) -> tuple[list[str], str]:  # (검색된 context 문서들, 모델 답변)
    context_docs = retrieve(question)
    context = "\n".join(f"- {d}" for d in context_docs)

    prompt = f"""아래 [사내 규정]만 근거로 질문에 답하라. 규정에 없는 내용은 "규정에서 확인할 수 없습니다"라고 답하라.

[사내 규정]
{context}

[질문]
{question}"""

    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return context_docs, res.choices[0].message.content

for q in [
    "긴급하게 발주해야 하는데 팀장이 자리를 비웠어. 어떻게 해?",
    "웨이퍼 재고가 400kg인데 자동 발주 알림이 와야 해?",  # 검색은 맞아도 "400 < 500" 숫자 비교를 모델이 틀릴 수 있음 (검색 정확 ≠ 추론 정확)
    "회사 주식 배당금은 언제 나와?",  # 규정 어디에도 없는 질문 → "규정에서 확인할 수 없습니다"가 나와야 정상
]:
    docs, a = answer(q)
    print(f"\n질문: {q}")
    print("검색된 근거:")
    for d in docs:
        print(f"  - {d}")
    print("답변:", a)
