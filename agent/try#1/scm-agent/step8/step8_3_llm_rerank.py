"""
Step 8-3: "임베딩으로 넉넉히 추리고(coarse retrieve) → LLM이 정밀 재정렬(rerank)"하는 2단계 검색.

step5_3(임베딩으로 후보 10개 중 상위 몇 개를 추려 LLM 재순위화; 4개 후보 버전은 step5_2)의 구조를 실제 검색 파이프라인 규모로 확장한 버전이다.
1단계(coarse_retrieve)는 빠르고 저렴한 임베딩으로 top_n(=6)개를 "넉넉히"(정답을 놓치지 않게) 뽑고,
2단계(llm_rerank)는 그중 소수만 LLM에게 "실제로 읽고 판단"시켜 최종 top_k(=3)로 정밀하게 줄인다.
LLM 호출이 느리고 비싸기 때문에 전체 문서가 아니라 1단계로 추려진 소수에만 쓰는 게 핵심 —
"싼 걸로 많이 거르고, 비싼 걸로 조금만 정확하게"라는 이 구조가 실무 RAG의 표준 패턴이다.
"""
import sys, os
import numpy as np
from openai import OpenAI
from pydantic import BaseModel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "step6"))
from docs import SCM_DOCS

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
CHAT_MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"

def embed(text: str) -> np.ndarray:
    return np.array(client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("문서 임베딩 중...")
DOC_VECTORS = [embed(d) for d in SCM_DOCS]

def coarse_retrieve(query: str, top_n: int = 6) -> list[str]:
    # 1단계: 임베딩으로 넉넉하게 후보를 뽑는다 (재현율 위주, 빠르고 저렴함).
    query_vec = embed(query)
    scored = sorted(((cosine_similarity(query_vec, v), d) for v, d in zip(DOC_VECTORS, SCM_DOCS)), reverse=True)
    return [d for _, d in scored[:top_n]]

class RerankResult(BaseModel):
    ranked_indices: list[int]  # 관련도 높은 순으로 정렬한 후보 인덱스

def llm_rerank(query: str, candidates: list[str], top_k: int = 3) -> list[str]:
    # 2단계: LLM이 후보들을 실제로 "읽고" 질문과의 관련도로 재정렬한다 (정확도 위주, 느리고 비쌈).
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates))
    prompt = f"""질문과 가장 관련 있는 순서대로 후보 문서의 번호를 정렬하라.

[질문]
{query}

[후보 문서]
{numbered}"""
    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_schema", "json_schema": {"name": "rerank", "schema": RerankResult.model_json_schema()}},
    )
    result = RerankResult.model_validate_json(res.choices[0].message.content)
    return [candidates[i] for i in result.ranked_indices[:top_k] if i < len(candidates)]

query = "거래처가 얼마나 믿을만한지 어떻게 판단해?"
candidates = coarse_retrieve(query)

print(f"\n질문: {query}")
print("\n1단계 (임베딩 top-6):")
for c in candidates:
    print(f"  - {c}")

print("\n2단계 (LLM 재랭킹 top-3):")
for c in llm_rerank(query, candidates):
    print(f"  - {c}")
