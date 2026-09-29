"""
Step 8-2: BM25(키워드)와 임베딩(의미)을 alpha 가중치로 섞은 하이브리드 검색.

step8_1에서 본 BM25의 약점(패러프레이즈에 취약)과 step5~7의 임베딩을 하나로 합쳐서
"단어가 겹치는 것"과 "의미가 비슷한 것" 둘 다를 반영하는 점수를 만든다.
alpha=1이면 순수 임베딩, alpha=0이면 순수 BM25 — 상황(도메인 전문용어가 많은지, 패러프레이즈가 많은지)에
따라 실측으로 튜닝해야 하는 값이다.
"""
import sys, os
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "step6"))
from docs import SCM_DOCS

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
EMBED_MODEL = "bge-m3"

bm25 = BM25Okapi([d.split() for d in SCM_DOCS])

def embed(text: str) -> np.ndarray:
    return np.array(client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("문서 임베딩 중...")
DOC_VECTORS = [embed(d) for d in SCM_DOCS]

def normalize(scores: np.ndarray) -> np.ndarray:
    # 스케일이 서로 다른 BM25 점수와 코사인 유사도를 0~1로 맞춰야 더할 수 있다.
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)

def hybrid_search(query: str, top_k: int = 3, alpha: float = 0.5):
    bm25_scores = normalize(np.array(bm25.get_scores(query.split())))
    query_vec = embed(query)
    embed_scores = normalize(np.array([cosine_similarity(query_vec, v) for v in DOC_VECTORS]))

    # 두 점수를 0~1로 정규화했기 때문에 이렇게 단순 가중합만 해도 스케일이 맞는다.
    combined = alpha * embed_scores + (1 - alpha) * bm25_scores
    ranked = sorted(zip(combined, SCM_DOCS), key=lambda x: x[0], reverse=True)
    return ranked[:top_k]

query = "거래처가 얼마나 믿을만한지 어떻게 판단해?"
print(f"\n질문: {query}")
for score, doc in hybrid_search(query):
    print(f"{score:.3f}  {doc}")

print("\n→ alpha가 1이면 순수 임베딩, 0이면 순수 BM25다. alpha=0.5로 섞으면 의미 검색과 키워드 검색을 절충한다.")
