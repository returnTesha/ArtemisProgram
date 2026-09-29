"""
Step 7-2: step7_1에서 저장해둔 ChromaDB 컬렉션을 검색 + 메타데이터 필터링으로 조회한다.

get_collection()은 이미 만들어진 컬렉션을 "다시 여는" 것 — 이 스크립트를 실행하기 전에
step7_1을 먼저 돌려서 데이터를 저장해둬야 한다 (없으면 에러).
where 조건으로 "category='재고'인 문서만"처럼 필터링하는 게 이 파일의 핵심 —
메모리 리스트(step6)라면 파이썬 for문으로 직접 걸러야 했을 일을 DB가 대신 해준다.
"""
import os
import chromadb
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
EMBED_MODEL = "bge-m3"

CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_store")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name="scm_docs")

def embed(text: str) -> list[float]:
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def search(query: str, top_k: int = 3, category: str | None = None):
    where = {"category": category} if category else None
    result = collection.query(
        query_embeddings=[embed(query)],
        n_results=top_k,
        where=where,  # None이면 필터 없이 전체 검색, dict를 주면 그 조건에 맞는 문서 중에서만 유사도 검색
    )
    # ChromaDB는 distance(거리)를 반환한다 — step5/6의 cosine similarity와 반대로 "작을수록 더 유사"하다.
    return list(zip(result["documents"][0], result["distances"][0], result["metadatas"][0]))

print("=== 전체 검색 ===")
for doc, dist, meta in search("발주 승인 절차가 궁금해"):
    print(f"[{meta['category']}] (거리 {dist:.4f}) {doc}")

print("\n=== category='재고' 로 필터링한 검색 ===")
for doc, dist, meta in search("자동 알림이 언제 발생해?", category="재고"):
    print(f"[{meta['category']}] (거리 {dist:.4f}) {doc}")

print("\n→ 메모리 리스트로는 '카테고리가 재고인 것만' 같은 조건 검색을 직접 필터링 코드로 짜야 하지만,")
print("   벡터DB는 where 조건 하나로 처리한다. 문서가 많아질수록 이 차이가 커진다.")
