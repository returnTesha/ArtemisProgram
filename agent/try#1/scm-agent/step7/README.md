# Step 7 — 벡터DB (ChromaDB)

Step 6의 "파이썬 리스트에 벡터 들고 있기"를 실제 벡터DB로 옮긴다. 영속성, 메타데이터 필터링, 대량 검색이 가능해진다.

## 실행
```
../.venv/bin/python step7_1_chromadb_setup.py   # 문서를 임베딩해서 ChromaDB에 저장
../.venv/bin/python step7_2_chromadb_query.py   # 검색 + category 필터링
../.venv/bin/python step7_3_rag_pipeline.py     # ChromaDB 기반 RAG 전체 파이프라인
```

## 관찰 포인트
- `step7_1_chromadb_setup.py`를 두 번 실행해도 문서가 중복 저장되지 않는다 (`upsert` 덕분) — 실행 후 `chroma_store/` 폴더가 생기는 것도 확인해보자.
- `step7_2_chromadb_query.py`는 **서로 다른 두 질문**을 실행한다: 필터 없는 검색("발주 승인 절차가 궁금해")과 `category='재고'` 필터 검색("자동 알림이 언제 발생해?"). 같은 질문을 필터 유무로 비교하는 게 아니므로, 결과 차이는 필터 효과만이 아니라 질문 차이도 섞여 있다. 그래도 `where` 한 줄로 조건 검색이 끝나는 점에서, 조건 검색이 왜 벡터DB에서 훨씬 쉬운지 체감할 수 있다.
- **`where` 필터의 핵심 (post-filter recall 함정, step5_3과 같은 구조):** ChromaDB의 `where`는 "category='재고'인 문서들 중에서만" 먼저 걸러낸 뒤 그 안에서 top_k를 뽑는다. 반대로 "먼저 top_k=3으로 뽑고 나서 Python에서 category=='재고'로 거르는" 순서로 짜면, 조건에 맞는 문서가 top_k 밖에 있을 때 결과가 0개가 될 수 있다. 다만 `step7_2`는 이 post-filter 방식을 실제로 실행해 보여주지는 않는다 — 위는 개념 설명이다. **태깅 주의:** `step7_1`은 문서에 "발주"가 들어 있으면 '발주', 아니면 "재고"가 들어 있으면 '재고', 아니면 '기타' 순서로 태깅한다. 그래서 "자동 알림이 언제 발생해?"에 실제로 답하는 문서(doc_2, "자동 발주 알림" 포함)는 '재고'가 아니라 '발주'로 태깅되고, '재고'로 태깅되는 문서는 doc_6, doc_8 두 개뿐이다. 따라서 `category='재고'` 필터 검색은 정답 문서(doc_2)를 애초에 후보에서 제외한다. 실제 서비스라면 미리 정리된 라벨을 써야 한다는 뜻이다.
- **실전 노트**: 처음엔 ChromaDB 경로를 `"./chroma_store"`(상대경로)로 뒀다가, 다른 폴더(step18)에서 이 파일을 import하니 엉뚱한 위치에 DB를 새로 만들려다 깨졌다. `__file__` 기준 절대경로로 고쳤다 — 경로는 항상 "어디서 실행되든" 기준으로 짜야 한다.
- **실전 노트 2 (거리 값이 1을 넘어서 당황할 수 있음)**: `step7_2`를 처음 돌렸을 때 "거리(distance)"가 1.15 같은 값까지 나와서 "코사인 유사도는 -1~1 아니었나?"라는 의문이 들 수 있다. 원인은 ChromaDB가 컬렉션 생성 시 `hnsw:space`를 안 정해주면 **기본값으로 코사인이 아니라 squared L2(유클리드) 거리**를 쓰기 때문이다 — step5/6에서 배운 코사인과 아예 다른 척도라 범위가 안 맞는 게 당연했다. `get_or_create_collection(..., metadata={"hnsw:space": "cosine"})`로 명시해서 고쳤다. 이 설정은 컬렉션 "생성 시점"에만 적용되므로, 이미 다른 space로 만들어진 기존 `chroma_store/`는 지우고 다시 만들어야 반영된다.

---

## English Summary

Moves step 6's "vectors in a python list" into a real vector database. You get persistence, metadata filtering, and search at scale.

**Run**
```
../.venv/bin/python step7_1_chromadb_setup.py   # embed documents and store them in ChromaDB
../.venv/bin/python step7_2_chromadb_query.py   # search + filter by category
../.venv/bin/python step7_3_rag_pipeline.py     # full RAG pipeline backed by ChromaDB
```

**What to notice**
- Run `step7_1_chromadb_setup.py` twice — documents aren't duplicated (thanks to `upsert`). Check that a `chroma_store/` folder appears.
- `step7_2_chromadb_query.py` runs **two different queries**: an unfiltered search ("발주 승인 절차가 궁금해") and a `category='재고'`-filtered search ("자동 알림이 언제 발생해?"). It does not compare the same query with and without the filter, so the difference in results mixes filter effects with query differences. Still, one `where` argument handles the conditional search, which shows why it's much easier with a vector DB.
- **The point of `where` (the post-filter recall trap, same structure as step5_3):** ChromaDB's `where` first restricts to documents with `category='재고'`, then picks top_k within that set. If you instead grab top_k=3 first and filter in Python afterwards, you can end up with zero results whenever the matching documents fall outside the top_k. Note that `step7_2` does not actually run that post-filter variant — this is a conceptual explanation. **Tagging caveat:** `step7_1` tags a document '발주' if it contains "발주", else '재고' if it contains "재고", else '기타'. So the document that actually answers "when does the auto-alert trigger?" (doc_2, which contains "자동 발주 알림") is tagged '발주', not '재고'; only doc_6 and doc_8 are tagged '재고'. The `category='재고'` filtered search therefore excludes the real answer (doc_2) from the candidates entirely — real services should use curated labels.
- **Real bug found**: the Chroma path was originally `"./chroma_store"` (relative). Importing this file from another folder (step18) broke it, since it tried to create a new DB in the wrong place. Fixed by resolving the path relative to `__file__` — paths should always work regardless of where the script is run from.
- **Real bug found #2 (distances above 1 are confusing at first)**: the first run of `step7_2` printed distances like 1.15, which looks wrong if you're expecting cosine similarity's -1..1 range. The cause: ChromaDB defaults to **squared L2 (Euclidean) distance**, not cosine, unless you set `hnsw:space` when creating the collection — a completely different scale than the cosine similarity taught in step5/6. Fixed with `get_or_create_collection(..., metadata={"hnsw:space": "cosine"})`. This setting only applies at collection *creation* time, so an existing `chroma_store/` built under the old default has to be deleted and rebuilt for it to take effect.
