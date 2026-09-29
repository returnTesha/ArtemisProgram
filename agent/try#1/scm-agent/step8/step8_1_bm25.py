"""
Step 8-1: 임베딩 검색과 정반대 성격인 키워드 검색(BM25)의 강점/한계를 확인한다.

step5~7까지는 임베딩(의미 기반) 검색만 다뤘다. 이 파일은 그 대척점에 있는 방식을 보여준다:
BM25는 "질문과 문서에 같은 단어가 얼마나 겹치는가"만으로 점수를 매기며, 의미 이해가 전혀 없다.
그래서 단어가 정확히 일치하는 질문엔 강하지만(품목 코드, 고유명사 등), 같은 뜻이라도
표현이 다르면(패러프레이즈) 점수가 0에 가까워진다 — 임베딩이 필요한 이유를 반대편에서 보여주는 예시.
"""
import sys, os
from rank_bm25 import BM25Okapi
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "step6"))
from docs import SCM_DOCS

# BM25는 "단어가 얼마나 겹치는가"로 점수를 매기는 키워드 검색이다. 임베딩과 달리 의미는 모른다.
tokenized_docs = [doc.split() for doc in SCM_DOCS]
bm25 = BM25Okapi(tokenized_docs)

def bm25_search(query: str, top_k: int = 3):
    scores = bm25.get_scores(query.split())
    ranked = sorted(zip(scores, SCM_DOCS), key=lambda x: x[0], reverse=True)
    return ranked[:top_k]

print("=== 일부 단어(협력사, 평가)가 그대로 겹치는 질문 (BM25가 상대적으로 유리) ===")
for score, doc in bm25_search("협력사 평가 등급 A등급 기준"):
    print(f"{score:.3f}  {doc}")

print("\n=== 같은 의미인데 단어가 다른 질문 (BM25가 취약함) ===")
for score, doc in bm25_search("거래처가 얼마나 믿을만한지 어떻게 판단해?"):
    print(f"{score:.3f}  {doc}")

print("\n→ '거래처'='협력사', '믿을만한지'='평가 등급' 처럼 단어가 다르면 BM25 점수가 확 낮아진다.")
print("   이게 임베딩 검색(의미 기반)이 필요한 이유이고, 반대로 품목 코드/고유명사처럼 정확 일치가 중요할 땐 BM25가 더 강하다.")
