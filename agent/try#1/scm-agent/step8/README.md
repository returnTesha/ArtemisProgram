# Step 8 — 검색 품질 향상

임베딩 검색이 항상 정답은 아니다. 키워드 검색(BM25)과 섞거나, LLM으로 다시 순위를 매기면 정확도가 올라간다.

## 실행
```
../.venv/bin/python step8_1_bm25.py           # 키워드 기반 검색의 강점과 한계
../.venv/bin/python step8_2_hybrid.py         # BM25 + 임베딩을 alpha로 섞기
../.venv/bin/python step8_3_llm_rerank.py      # 임베딩으로 넉넉히 뽑고 LLM이 재정렬
../.venv/bin/python step8_4_full_pipeline.py  # 하이브리드(BM25+임베딩) 1차 검색 + LLM 재정렬을 하나로 합친 완성형
```

## 관찰 포인트
- `step8_1_bm25.py`에서 "거래처가 얼마나 믿을만한지"처럼 단어가 다른 질문의 점수가 전부 0으로 나오는 걸 확인 — 이게 임베딩 검색이 필요한 이유다.
- `step8_2_hybrid.py`의 `hybrid_search(query)`는 기본값 `alpha=0.5`로 한 번만 호출된다. `alpha`를 0, 0.5, 1 등으로 직접 바꿔 다시 실행하며 결과가 어떻게 달라지는지 실험해보자.
- `step8_3_llm_rerank.py`의 1단계(임베딩 top-6)와 2단계(LLM 재정렬 top-3) 순서가 실제로 바뀌는 걸 확인 — 재랭킹이 느리지만 왜 필요한지 체감할 수 있다.
- **`step8_1`~`step8_3`은 서로 이어지는 파이프라인이 아니다.** 특히 `step8_3`의 1단계(`coarse_retrieve`)는 순수 임베딩만 쓰고 BM25는 전혀 안 쓴다 — `step5_3`의 구조를 규모만 키운 것에 가깝다. `step8_4_full_pipeline.py`가 실제로 셋을 이어붙인 버전: 1단계를 `step8_2`와 같은 로직의 하이브리드 검색(BM25+임베딩)으로 바꾸고, 그 결과를 `step8_3`과 같은 로직의 `llm_rerank`에 넘긴다 (import가 아니라 `coarse_retrieve`/`normalize`/`llm_rerank`를 이 파일에 복사해 둔 것).
- `step8_4`를 두 질문으로 돌려서 비교해볼 것: "거래처가 얼마나 믿을만한지"(패러프레이즈, 임베딩이 강한 영역)와 "위험물 단독 검수 규정 위반이야?"(일부 키워드가 겹침, BM25가 유리한 쪽). BM25는 공백 `.split()`으로 토큰화하므로 "위험물" vs "위험물(불산", "검수" vs "검수를"처럼 조사/괄호가 붙은 단어는 매칭되지 않고 일부 토큰만 겹친다. 정답 문서가 하이브리드 1단계에서 몇 위에 오르는지(1위 보장은 없음) 직접 확인하고 순수 임베딩(`step8_3`)과 비교해보자.

---

## English Summary

Embedding search isn't always right. Mixing in keyword search (BM25) or re-ranking with an LLM can improve accuracy.

**Run**
```
../.venv/bin/python step8_1_bm25.py           # BM25's strengths and blind spots
../.venv/bin/python step8_2_hybrid.py         # blend BM25 + embeddings via alpha
../.venv/bin/python step8_3_llm_rerank.py      # cast a wide embedding net, then let an LLM re-rank
../.venv/bin/python step8_4_full_pipeline.py  # the complete version: hybrid (BM25+embedding) retrieve + LLM rerank, combined
```

**What to notice**
- In `step8_1_bm25.py`, a paraphrased query ("how trustworthy is this supplier?") scores 0 against every document — this is exactly why embedding search matters.
- `step8_2_hybrid.py` calls `hybrid_search(query)` just once, with the default `alpha=0.5`. Change `alpha` yourself (0, 0.5, 1, ...) and rerun to see how the ranking shifts.
- In `step8_3_llm_rerank.py`, watch the order actually change between the stage-1 (embedding top-6) and stage-2 (LLM re-ranked top-3) results.
- **`step8_1`–`step8_3` are not a connected pipeline.** In particular, `step8_3`'s stage 1 (`coarse_retrieve`) is pure embedding search — no BM25 at all — closer to `step5_3`'s pattern scaled up than to a hybrid pipeline. `step8_4_full_pipeline.py` is the version that actually chains all three: stage 1 becomes the same logic as `step8_2`'s hybrid search (BM25+embedding), and that output feeds the same logic as `step8_3`'s `llm_rerank` (not imported — `coarse_retrieve`/`normalize`/`llm_rerank` are copied into this file).
- Run `step8_4` against two queries and compare: "how trustworthy is this supplier?" (a paraphrase, embedding's strength) vs. "is unsupervised hazmat inspection against policy?" (partial keyword overlap, where BM25 has an edge). BM25 tokenizes with whitespace `.split()`, so words with attached particles/punctuation don't match ("위험물" vs "위험물(불산", "검수" vs "검수를") — only some tokens overlap. Check where the correct document lands in the hybrid stage-1 results (rank 1 is not guaranteed; verify by running) and compare against pure embedding retrieval (`step8_3`).
