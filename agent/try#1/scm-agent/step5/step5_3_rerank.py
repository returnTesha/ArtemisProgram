"""
Step 5-3: 임베딩(1차 필터) → LLM 재순위화(2차 정밀 판단)

지금까지 배운 두 가지 방식을 실제로 "조합"해서 실무 RAG 파이프라인의 축소판을 만든다.

  A) 임베딩만으로 유사도 계산  → 빠르고 저렴, 대량 데이터에 강함, 그러나 도메인 지식(system 프롬프트)을 못 넣음
  C) 처음부터 LLM에게 system 프롬프트로 순위 매기게 함 → 도메인 지식 반영 가능, 그러나 느리고 비쌈(후보가 많으면 못 씀)

  B) 이 파일이 하는 것: A로 대량 후보를 TOP_K개로 "추린 뒤", 그 소수만 C처럼 LLM에게 넘겨 재순위화한다.
     → 실무에서 "retrieve(임베딩) → rerank(LLM)"이라고 부르는 바로 그 구조.

주의: 이 구조는 은탄환이 아니다. TOP_K가 너무 작으면 1차 필터링 단계에서 도메인상 중요한 후보가
아예 탈락해버릴 수 있고, 그러면 2차 LLM이 아무리 똑똑해도 되살릴 수 없다 (아래 shortlist 부분의
주석 참고). "몇 개를 추릴지(TOP_K)"는 recall(놓치지 않기) vs 비용(LLM 호출 수) 사이의 실전 튜닝 포인트다.
"""

from typing import List
import numpy as np
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

# 실제 서비스에서는 후보가 수천~수백만 개일 수 있다.
# 그중 LLM에게 넘길 "소수"를 몇 개로 자를지 정하는 값 — 이게 바로 A와 B를 가르는 핵심 파라미터다.
TOP_K = 3


def embed(text: str):
    """텍스트를 벡터로 변환한다. 학습된 가중치로 하는 '고정된 계산'이라 판단/추론이 실시간으로 일어나지 않는다.
    → 그래서 이 함수엔 system 프롬프트를 넣을 자리가 아예 없다 (embeddings API는 messages/role 구조가 없음)."""
    res = client.embeddings.create(model="bge-m3", input=text)
    return res.data[0].embedding


def cosine_similarity(a, b):
    """두 벡터 사이 코사인 유사도. 여기엔 AI가 전혀 관여하지 않는다 — 순수 산수(내적/크기)일 뿐이다."""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


query = "창고에 재고가 없어요"  # 이건 "질문/문의"다 — 사용자가 지금 겪고 있는 문제.

# candidates는 "다른 사용자의 질문들"이 아니라 미리 확보해둔 "지식/사실(문서)"이어야 한다.
# (질문끼리 비교하는 건 "과거에 비슷한 문의가 있었나 찾기"용 다른 용도지, 지금 하려는 건
#  "이 문제에 관련된 사내 지식을 찾아서 답에 활용하기"이므로 후보는 지식 쪽이어야 맞다.)
# 여기 문장들은 step6/docs.py의 SCM_DOCS와 같은 성격의 "사내 규정/사실"이다 —
# 실제 RAG(step7)에서 벡터 DB에 미리 저장해두는 게 바로 이런 지식 문서들이다.
candidates = [
    "재고 안전 수량 기준: 웨이퍼 500kg, 포토레지스트 100kg, 불산 10kg 미만이면 자동 발주 알림이 발생한다.",  # 재고 부족 자체에 관한 규정 → 1순위 예상
    "협력사 납기 지연이 3일을 넘으면 재고 부족으로 이어질 수 있어 대체 협력사 소싱을 검토해야 한다.",  # 도메인 인과관계가 명시된 사실 → 임베딩은 낮게, LLM은 높게 볼 것
    "C사 납품 협력사는 최근 배송 지연 이력이 있어 모니터링 대상이다.",  # 위와 같은 유형(다른 협력사) — 일반화되는지 확인용
    "웨이퍼 입고 검수는 품질팀 입회 하에 진행해야 한다.",  # 재고/웨이퍼 관련이지만 "부족"과는 무관
    "입고 예정일은 사내 시스템 공지 게시판에서 확인할 수 있다.",  # 단어(입고)는 겹치지만 인과관계는 약함
    "포토레지스트 발주 수량은 지난 3개월 평균 사용량 기준으로 산정한다.",  # 다른 카테고리(발주)
    "창고 내 위험물(불산 등) 보관은 별도 구역에 분리해야 한다.",  # "창고"라는 단어만 겹치는 무관한 규정
    "사내 구내식당 점심 메뉴는 매주 월요일에 공지된다.",  # 완전히 무관
    "다음 주 화요일 ERP 시스템 점검으로 발주 시스템 접속이 2시간 제한된다.",  # 완전히 무관
    "회의실 예약은 그룹웨어 시스템에서 진행한다.",  # 완전히 무관
]

# ============================================================
# 1차: 임베딩으로 전체 후보를 훑어서 상위 TOP_K개만 추려낸다 (Retrieve)
# ============================================================
# 실무라면 이 단계는 벡터 DB(step7의 ChromaDB)에 미리 저장해둔 수천~수백만 개 벡터를 대상으로
# "가장 가까운 이웃 찾기(ANN)"를 하는 부분이다. 여기선 개념만 보여주려고 10개를 다 계산한다.
print(f"=== 1차: 임베딩으로 전체 {len(candidates)}개 후보 훑기 (Retrieve) ===")
query_vec = embed(query)
embed_scores = [(c, cosine_similarity(query_vec, embed(c))) for c in candidates]
embed_scores.sort(key=lambda x: -x[1])
for c, s in embed_scores:
    print(f"  {s:.4f}  {c}")

# 여기서 실제로 "추린다" — 상위 TOP_K개만 다음 단계로 넘기고 나머지는 버린다.
# (후보가 10,000개였다면 이 한 줄이 "10,000번 LLM 호출"을 "TOP_K번 LLM 호출"로 줄여주는 진짜 이유다.)
#
# 주의(실제로 겪는 위험 — TOP_K=3 기준 실측): "창고 내 위험물 보관" 규정이 "창고"라는 단어만
# 겹칠 뿐 지금 문제(재고 부족)와 사실상 무관한데도 임베딩 1위(0.5677)를 차지하고, 정작 원인 관계가
# 있는 "협력사 납기 지연이 3일을 넘으면 재고 부족으로..." 규정은 근소한 차이로 2위(0.5270)에 그친다.
# 다행히 TOP_K=3이라 둘 다 살아남아서 2차 LLM이 순서를 바로잡을 기회는 있지만, 6위(0.4309)로 밀린
# "C사 납품 협력사는 최근 배송 지연 이력이 있어..." 같은 다른 지식은 컷에서 탈락해 2차 단계에
# 아예 전달되지 않는다 — 2차 LLM이 아무리 똑똑해도 이미 걸러져서 넘어오지 않은 후보는 되살릴
# 수 없다 (rerank는 "이미 뽑힌 것들의 순서"만 고칠 뿐 "애초에 안 뽑힌 것"은 못 구한다).
# TOP_K를 1로 낮추면 핵심 규정마저 통째로 탈락하는 걸 직접 실험해볼 것 — TOP_K를 몇으로 잡을지는
# recall(필요한 지식을 놓치지 않기) vs 비용(LLM에 몇 개나 넘길지) 사이의 실전 튜닝 문제다.
shortlist = [c for c, _ in embed_scores[:TOP_K]]
print(f"\n  → 상위 {TOP_K}개만 다음 단계로 추림: {shortlist}")

# ============================================================
# 2차: 추려진 소수만 LLM에게 넘겨서 system 프롬프트로 재순위화 (Rerank)
# ============================================================
# chat completions API는 embeddings API와 달리 messages=[{role: system}, ...] 구조가 있어서
# "지금 이 순간" 도메인 지식을 지시로 끼워넣을 수 있다 — 임베딩 단계에선 불가능했던 일.
class RelevanceItem(BaseModel):
    text: str
    relevance: int  # 0~10, 도메인 지식을 반영한 관련도
    reason: str


class RerankResult(BaseModel):
    items: List[RelevanceItem]


SYSTEM = """너는 반도체 SCM 문의에 참고할 사내 지식 문서를 재평가하는 도우미다.
사용자 문의와 검색된 지식 문서 후보들이 주어지면, SCM 도메인 지식을 바탕으로 각 문서의 관련도(0~10)를 매겨라.

[중요한 도메인 지식]
- "재고가 없다/부족하다"의 흔한 원인은 "협력사 납기 지연"이다. 표면적 단어가 달라도 인과관계로 연결되면 관련도를 높게 줘라.
- 완전히 무관한 화제(점심, 회의실 예약 등)는 관련도를 낮게 줘라."""

shortlist_text = "\n".join(f"- {c}" for c in shortlist)
res = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"문의: {query}\n\n지식 문서 후보:\n{shortlist_text}"},
    ],
    temperature=0,  # 재현 가능한 결과를 보려고 무작위성을 끔 (step4에서 본 것처럼 온도가 높으면 결과가 매번 달라진다)
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "rerank", "schema": RerankResult.model_json_schema()},
    },
)
result = RerankResult.model_validate_json(res.choices[0].message.content)

print(f"\n=== 2차: 추려진 {len(shortlist)}개만 LLM 재순위화 (Rerank, system 프롬프트로 도메인 지식 주입) ===")
for item in sorted(result.items, key=lambda x: -x.relevance):
    print(f"  {item.relevance:2d}  {item.text}  ({item.reason})")

# 관찰 포인트: TOP_K=3 기본값으로 돌리면 "창고 내 위험물 보관" 규정이 "창고"라는 단어만 겹쳐서
# 임베딩 1위를 차지했다가, 2차 LLM 재순위화에서 관련도 0점으로 정확히 강등되고, 진짜 원인 관계가
# 있는 "협력사 납기 지연..." 규정이 1위로 올라간다 — rerank가 "단어 유사성"과 "실제 관련성"을
# 구분해내는 걸 직접 확인할 수 있다. 다만 "C사 납품 협력사는 최근 배송 지연 이력이..." 같은,
# 역시 참고할 만한 지식은 TOP_K=3 컷에서 아예 탈락해 2차 결과에 나타나지도 않는다 — TOP_K를
# 낮춰가며(예: 1) 어느 지점에서 핵심 지식마저 통째로 놓치는지 실험해볼 것.
