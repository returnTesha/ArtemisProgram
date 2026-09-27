# Step 5 — Embedding

텍스트를 숫자 벡터로 바꾸면, 벡터 사이의 거리(코사인 유사도)로 "의미가 비슷한지"를 계산할 수 있다. RAG의 기초.

## 실행
```
../.venv/bin/python step5_1_embedding.py    # 텍스트 하나를 벡터로 변환해서 차원 수 확인
../.venv/bin/python step5_2_similarity.py   # 질문과 여러 후보 문장의 유사도 순위 매기기
../.venv/bin/python step5_3_rerank.py       # 임베딩 순위 vs LLM 재순위화(rerank) 비교
```

## 관찰 포인트
- `step5_1_embedding.py`의 벡터 차원 수(bge-m3는 1024차원)를 확인해보자 — 이 숫자가 이후 스텝에서 임베딩을 다룰 때 계속 등장한다.
- `step5_2_similarity.py`에서 "창고에 재고가 없어요"와 "재고가 부족해요"의 유사도가 "오늘 점심 메뉴 추천해줘"보다 훨씬 높게 나오는지 확인 — 단어가 안 겹쳐도 의미로 판단된다는 게 핵심이다.
- 후보 문장을 직접 추가해서 순위가 예상대로 나오는지 실험해보면 감이 빨리 잡힌다.
- **임베딩의 한계:** `step5_2`를 자세히 보면 "B 협력사 납기가 지연됐어요"(0.5361)와 "오늘 점심 메뉴 추천해줘"(0.4901)의 점수 차이가 겨우 0.05다. SCM 관점에선 "납기 지연"이 "재고 부족"의 흔한 **원인**이라 훨씬 관련도가 높아야 하는데, `bge-m3`는 표면적 단어 유사성만 볼 뿐 이런 도메인 인과관계는 모른다. **임베딩 API엔 `system` 프롬프트를 넣을 자리 자체가 없어서(=messages/role 구조가 없음) 이 문제는 프롬프트로 못 고친다.**
- **중요: 무엇을 임베딩해서 저장해둬야 하는가.** `step5_3`의 candidates는 "다른 사용자의 질문들"이 아니라 미리 확보해둔 **지식/사실(문서)**이다 — `step6/docs.py`의 `SCM_DOCS`("납기 지연이 3일을 넘으면 대체 협력사 소싱을 검토해야 한다" 같은 사내 규정)와 같은 성격. 질문끼리 비교하는 건 "과거에 비슷한 문의가 있었나 찾기"(FAQ/티켓 라우팅)용 다른 용도이고, 지금 하려는 RAG는 "이 문의에 관련된 사내 지식을 찾아서 답에 활용하기"이므로 임베딩해서 쌓아두는 대상은 지식 쪽이어야 맞다.
- `step5_3_rerank.py`는 이 지식 문서 10개를 임베딩으로 훑어서 **상위 `TOP_K`개만 추려낸 뒤(줄인 뒤)**, 그 소수만 **채팅 모델**에 넘겨 `system` 프롬프트로 도메인 지식("재고 부족의 흔한 원인은 납기 지연이다")을 주입해서 재순위화(rerank)한다 — RAG 파이프라인에서 "retrieve(임베딩, 대량→소수) → rerank(LLM, 소수→정밀 순위)"이라 부르는 흔한 2단계 구조. 실행해보면 "창고 내 위험물 보관" 규정이 "창고"라는 단어만 겹쳐서 임베딩 1위(0.5677)를 차지했다가, 2차 LLM이 관련도 0점으로 정확히 강등시키고 진짜 원인 관계가 있는 "협력사 납기 지연..." 규정을 1위로 올리는 걸 볼 수 있다.
- **더 중요한 반전(TOP_K를 너무 작게 잡으면 실패한다):** 위 사례는 다행히 TOP_K=3 안에 핵심 규정이 살아남아서 rerank가 순서를 바로잡을 수 있었지만, "C사 납품 협력사는 최근 배송 지연 이력이 있어..." 같은 역시 참고할 만한 지식은 6위(0.4309)라 TOP_K=3 컷에서 **아예 탈락**해 2차 결과에 나타나지도 않는다. 2차 LLM이 아무리 똑똑해도 이미 걸러져서 넘어오지 않은 후보는 되살릴 수 없다 — rerank는 "이미 뽑힌 것들의 순서"만 고칠 뿐 "애초에 안 뽑힌 것"은 못 구한다. TOP_K를 1로 낮추면 핵심 규정마저 통째로 탈락하는 걸 직접 확인할 수 있다. 즉 **TOP_K를 몇으로 잡을지는 recall(필요한 지식을 놓치지 않기)과 비용(LLM에 몇 개나 넘길지) 사이의 실전 튜닝 문제**이지, 한 번 정해두면 끝나는 값이 아니다.
- **정리(비용/난이도 스펙트럼):** ① 프롬프트 엔지니어링(즉시, 무료, chat 모델에만 적용 가능) → ② RAG/파이프라인 구성(오늘 한 rerank, 코드만 추가하면 됨, 학습 불필요) → ③ 파인튜닝(모델 가중치 자체를 재학습, 데이터+GPU 필요, 비용이 훨씬 큼). 중소규모 팀은 보통 ①②로 충분하고, ③은 트래픽이 아주 크거나 독점 데이터가 쌓였을 때만 고려하는 최후의 수단이다.

---

## English Summary

Turning text into vectors lets you compute "semantic similarity" via cosine distance. This is the foundation of RAG.

**Run**
```
../.venv/bin/python step5_1_embedding.py    # embed one text, check the vector's dimensionality
../.venv/bin/python step5_2_similarity.py   # rank candidate sentences by similarity to a query
../.venv/bin/python step5_3_rerank.py       # embedding ranking vs. LLM rerank, side by side
```

**What to notice**
- Note the embedding dimension (bge-m3 is 1024) — it reappears constantly in later RAG steps.
- In `step5_2_similarity.py`, "no stock in the warehouse" should score much higher against "stock is low" than against "recommend lunch" — despite sharing almost no words, semantic meaning wins.
- Add your own candidate sentences and see if the ranking matches your intuition.
- **A real limit of embeddings:** look closely at `step5_2` — "B supplier's delivery is delayed" (0.5361) and "recommend lunch" (0.4901) are only 0.05 apart. From an SCM standpoint, delivery delay is a common *cause* of low stock and should score much higher, but `bge-m3` only sees surface-level word similarity, not domain causality. **The embeddings API has no place to put a `system` prompt at all (no messages/role structure), so this can't be fixed with prompting.**
- **Important: what should you actually embed and store?** The candidates in `step5_3` are not "other users' questions" — they're pre-collected **knowledge/facts (documents)**, the same kind of thing as `SCM_DOCS` in `step6/docs.py` (internal policy statements like "if delivery is delayed more than 3 days, review alternate suppliers"). Comparing question-to-question is a different use case (FAQ / duplicate-ticket matching); real RAG here means "find internal knowledge relevant to this inquiry," so what you embed and store ahead of time has to be the knowledge side, not more questions.
- `step5_3_rerank.py` scans 10 such knowledge documents with embeddings, **narrows down to the top `TOP_K`**, then feeds only that shortlist into a **chat model** and reranks it with a `system` prompt that injects domain knowledge ("low stock is often caused by supplier delivery delay") — the classic RAG "retrieve (embedding, many→few) → rerank (LLM, few→precise order)" two-stage pattern. Run it and watch the "keep hazardous materials in a separate warehouse zone" policy grab embedding rank #1 (0.5677) purely because it shares the word "warehouse," while the LLM correctly demotes it to 0 and promotes the genuinely causal "supplier delivery delay" policy to #1.
- **A more important twist (too small a `TOP_K` breaks the whole thing):** that case got lucky — the causal policy survived the `TOP_K=3` cut, so rerank had a chance to fix the order. But another useful fact, "supplier C has a recent history of delayed shipments," sits at rank 6 (0.4309) and gets **cut entirely** — it never reaches the LLM rerank stage at all. No matter how smart that stage is, it can't rescue a candidate that never made it through. Try `TOP_K=1` and watch even the critical policy get dropped entirely. **Picking `TOP_K` is a real recall-vs-cost tuning problem**, not a value you set once and forget.
- **The cost/effort spectrum:** ① prompt engineering (instant, free, only works on chat models) → ② RAG/pipeline design (today's rerank — just add code, no training) → ③ fine-tuning (actually retrain the model's weights — needs data + GPU, far more expensive). Small-to-mid teams usually stop at ①/②; ③ is a last resort reserved for very high traffic or genuinely proprietary data.
