# Step 18 — 평가 (Evaluation)

정답이 하나로 정해지지 않는 자유 서술형 답변을, LLM을 "심사위원"으로 써서 자동으로 채점한다.

## 실행
```
../.venv/bin/python step18_1_llm_judge.py       # 질문/기준답변/실제답변을 pass·fail로 채점
../.venv/bin/python step18_2_eval_pipeline.py   # step7의 RAG 파이프라인 전체를 여러 케이스로 평가
```

## 관찰 포인트
- `step18_1_llm_judge.py`에서 표현이 다른데도 핵심 사실이 맞으면 pass로 판정되는지 확인해보자 (정확히 같은 문장이 아니어도 된다). 같은 파일의 오답(`500kg`)·애매한 답 케이스가 fail로 나오는지도 함께 봐야 심사위원이 제대로 작동하는 것이다.
- `step18_2_eval_pipeline.py`의 마지막 "통과율"을 확인 — RAG 파이프라인 전체(검색+생성)를 한 번에 평가하는 방식이다. 케이스 5개 중 마지막(같은 질문에 기준 답변 "불가능합니다")은 일부러 틀린 기준을 준 대조군이라 **fail이 나와야 정상**이다 — 그래서 정상 결과는 4/5(80%)이고, 5/5면 오히려 심사위원이 아무거나 pass를 주고 있다는 뜻이다. 케이스가 적어서 4/5여도 파이프라인이 좋다는 강한 증거는 아니다. 통과율이 낮아도 검색 문제인지 생성 문제인지는 이 결과만으론 구분할 수 없다.
- `step18_2`의 심사위원은 답변을 만드는 모델과 같은 `qwen2.5:7b`이고, `judge()`는 `step18_1`의 것을 import하지 않고 프롬프트를 줄여 다시 정의한 것이라 채점 기준이 완전히 같지는 않다.
- **실전 노트 (`step18_1`에서 겪은 일)**: 처음엔 `reasoning`, `verdict`, `score` 세 필드를 같이 물었는데, `reasoning`/`verdict`와 `score`가 서로 어긋나는 자기모순(예: 근거는 맞다고 해놓고 1점)이 나왔다. `score`를 없애고 `reasoning` + `verdict` 두 필드만 남기니 해결됐다 — LLM에게 여러 판단을 동시에 시키면 서로 어긋날 수 있다는 걸 보여주는 실제 사례다.

---

## English Summary

Free-form answers rarely have one "correct" wording. Use an LLM as a judge, with clear grading criteria, to score them automatically.

**Run**
```
../.venv/bin/python step18_1_llm_judge.py       # grade question/reference/answer triples as pass/fail
../.venv/bin/python step18_2_eval_pipeline.py   # evaluate step 7's full RAG pipeline on a test set
```

**What to notice**
- In `step18_1_llm_judge.py`, confirm a differently-worded answer still passes as long as the core fact is correct. Also check that the wrong answer (`500kg`) and the vague answer in the same file come out as fail — that's how you know the judge actually works.
- Check the final pass rate in `step18_2_eval_pipeline.py` — it evaluates the whole pipeline (retrieval + generation) at once. The 5th case (same question, reference "불가능합니다") is a deliberately wrong reference used as a negative control, so it **should fail** — the healthy result is 4/5 (80%), and 5/5 would mean the judge passes anything. With so few cases even 4/5 is weak evidence, and a low rate can't tell you whether retrieval or generation is at fault.
- The judge in `step18_2` is the same `qwen2.5:7b` that generates the answers, and its `judge()` is a re-defined, shorter-prompt copy rather than an import from `step18_1`, so the grading criteria are not identical.
- **Real bug found (in `step18_1`)**: asking for `reasoning`, `verdict`, and `score` together produced self-contradictions between `reasoning`/`verdict` and `score` — e.g., a reasoning that says the answer is right, paired with a score of 1. Dropping `score` and keeping `reasoning` + `verdict` fixed it. Asking an LLM to make several judgments at once invites them to disagree with each other.
