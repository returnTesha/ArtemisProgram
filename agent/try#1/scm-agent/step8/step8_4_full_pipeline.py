"""
Step 8-4: step8_1~8_3을 전부 합친 "완성형" 검색 파이프라인.

step8_1(BM25) / step8_2(하이브리드) / step8_3(임베딩→LLM rerank)는 각각 독립된 기법 하나씩만
보여줬다 — 서로 이어지는 파이프라인이 아니었다. 특히 step8_3의 1단계(coarse_retrieve)는
순수 임베딩만 썼지 BM25는 전혀 쓰지 않았다. 실무에서 가장 완성된 형태는 이 셋을 다음처럼 잇는 것이다:

  1단계 (coarse retrieve, 재현율 위주, 싸고 빠름):
    step8_2와 같은 로직의 하이브리드 검색(BM25 + 임베딩, alpha 가중합; import가 아니라 이 파일에 복사해 둠)으로 top_n개를 "넉넉히" 뽑는다.
    → 순수 임베딩만 쓰는 step8_3과 달리, 질문과 단어가 겹치는 문서(BM25가 강한 영역)도 후보에 반영할 수 있다.
  2단계 (rerank, 정밀도 위주, 비싸고 느림):
    step8_3과 같은 로직의 llm_rerank(역시 이 파일에 복사해 둠)로 그 top_n 중 진짜 관련 있는 top_k만 정밀하게 골라낸다.

"싼 방법 두 개(키워드+의미)를 합쳐서 넓게 훑고, 비싼 방법(LLM)은 그 소수에만 쓴다"는 게 핵심 —
이게 실무 검색/RAG 시스템이 실제로 쓰는 표준 구조(retrieve 단계 자체도 하이브리드, 그 위에 rerank)다.
"""
import sys, os
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI
from pydantic import BaseModel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "step6"))
from docs import SCM_DOCS

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
CHAT_MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"

# --- 1단계 재료: BM25(키워드) + 임베딩(의미) ---
bm25 = BM25Okapi([d.split() for d in SCM_DOCS])

def embed(text: str) -> np.ndarray:
    return np.array(client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("문서 임베딩 중...")
DOC_VECTORS = [embed(d) for d in SCM_DOCS]

def normalize(scores: np.ndarray) -> np.ndarray:
    # BM25 점수와 코사인 유사도는 스케일이 완전히 다르므로(BM25는 0~수십, 코사인은 -1~1),
    # 0~1로 맞춰야 가중합이 의미를 갖는다 (step8_2와 동일한 이유).
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)

def coarse_retrieve(query: str, top_n: int = 6, alpha: float = 0.5) -> list[str]:
    """1단계: BM25 + 임베딩을 alpha로 섞은 하이브리드 점수로 top_n개를 넉넉히 뽑는다 (재현율 위주)."""
    bm25_scores = normalize(np.array(bm25.get_scores(query.split())))
    query_vec = embed(query)
    embed_scores = normalize(np.array([cosine_similarity(query_vec, v) for v in DOC_VECTORS]))
    combined = alpha * embed_scores + (1 - alpha) * bm25_scores
    ranked = sorted(zip(combined, SCM_DOCS), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_n]]

# --- 2단계: LLM이 후보를 "읽고" 정밀하게 재정렬 (step8_3과 동일) ---
class RerankResult(BaseModel):
    ranked_indices: list[int]  # 관련도 높은 순으로 정렬한 후보 인덱스

def llm_rerank(query: str, candidates: list[str], top_k: int = 3) -> list[str]:
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

def full_pipeline(query: str, top_n: int = 6, top_k: int = 3) -> tuple[list[str], list[str]]:
    coarse = coarse_retrieve(query, top_n=top_n)
    fine = llm_rerank(query, coarse, top_k=top_k)
    return coarse, fine

for query in [
    "거래처가 얼마나 믿을만한지 어떻게 판단해?",  # step8_2/8_3와 동일한 질문 — 패러프레이즈(단어가 안 겹침)
    "위험물 단독 검수 규정 위반이야?",           # 키워드(위험물, 단독 등)가 문서와 일부 겹치는 질문 — BM25가 유리한 쪽 (공백 split 토큰화라 조사/괄호가 붙은 단어는 매칭 안 됨)
]:
    coarse, fine = full_pipeline(query)
    print(f"\n질문: {query}")
    print("1단계 (하이브리드 top-6):")
    for d in coarse:
        print(f"  - {d}")
    print("2단계 (LLM 재랭킹 top-3):")
    for d in fine:
        print(f"  - {d}")

# 관찰 포인트: 두 번째 질문("위험물 단독 검수...")은 일부 토큰만 문서와 겹친다. BM25는 공백 .split()으로 토큰화하므로
# "위험물" vs "위험물(불산", "검수" vs "검수를"처럼 조사/괄호가 붙은 단어는 매칭되지 않는다.
# 그래도 겹치는 토큰이 BM25 성분으로 기여해 정답 문서가 1단계 상위에 오르는지(1위 보장은 없음, 직접 확인 필요) 살펴볼 것.
# 순수 임베딩(step8_3)과 비교해 어떻게 달라지는지 보는 게 이 파일의 관찰 포인트다.
