"""
Step 7-1: step6의 "파이썬 리스트에 벡터 들고 있기"를 실제 벡터 DB(ChromaDB)로 옮긴다.

메모리 리스트 방식(step6)은 스크립트가 끝나면 임베딩이 사라져서 매번 재계산해야 하고,
조건 필터링(카테고리별 검색 등)도 직접 루프를 짜야 했다. ChromaDB는 이걸 디스크에 영속 저장하고,
메타데이터 필터(where 조건)와 대량 검색(수백만 건)을 DB 차원에서 지원한다.
"""
import sys, os
import chromadb
from openai import OpenAI
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "step6"))
from docs import SCM_DOCS

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
EMBED_MODEL = "bge-m3"

def embed_many(texts: list[str]) -> list[list[float]]:
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]

# __file__ 기준 절대경로를 쓰는 이유: 상대경로("./chroma_store")로 두면, 다른 폴더(예: step18)에서
# 이 모듈을 import했을 때 실행 위치 기준으로 엉뚱한 곳에 DB를 새로 만들려다 깨진다.
# → 경로는 "어디서 실행되든" 항상 같은 곳을 가리키도록 파일 위치 기준으로 짜야 한다는 실전 교훈.
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_store")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
# hnsw:space를 안 정해주면 ChromaDB는 기본값으로 코사인이 아니라 squared L2(유클리드 거리)를 쓴다.
# step5/6에서 배운 코사인 유사도와 값 범위가 안 맞아 헷갈리는 걸 막으려고 명시적으로 cosine을 지정한다.
# (컬렉션 생성 시점에만 적용되는 설정이라, 이미 다른 space로 만들어진 컬렉션은 삭제 후 재생성해야 바뀐다.)
collection = chroma_client.get_or_create_collection(name="scm_docs", metadata={"hnsw:space": "cosine"})

# 이미 저장돼 있으면 중복 저장하지 않는다 (id 기준 upsert) → 이 스크립트를 몇 번 다시 돌려도 문서가 안 늘어남.
ids = [f"doc_{i}" for i in range(len(SCM_DOCS))]
# 문서 내용에 특정 단어가 있는지로 카테고리를 대충 태깅 (진짜 서비스라면 미리 정리된 라벨을 써야 함).
# 이 metadata가 step7_2의 where 필터링에 쓰인다.
metadatas = [{"category": "발주" if "발주" in d else "재고" if "재고" in d else "기타"} for d in SCM_DOCS]

collection.upsert(
    ids=ids,
    documents=SCM_DOCS,
    embeddings=embed_many(SCM_DOCS),
    metadatas=metadatas,
)

print(f"저장된 문서 수: {collection.count()}")
print("샘플 조회:", collection.get(ids=["doc_0"]))
