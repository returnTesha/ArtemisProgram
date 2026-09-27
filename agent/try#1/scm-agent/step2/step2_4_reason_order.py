"""
Step 2-4: 같은 정보인데 필드 "순서"만 다른 두 스키마(ReasonFirst vs ReasonLast)를 비교.

이 스텝의 핵심 주장: 스키마는 단순 검증 도구가 아니라 모델의 추론 순서를 설계하는 레버다.
reason 필드를 category/urgency보다 앞에 두느냐 뒤에 두느냐가
"먼저 생각하고 결론 내리기" vs "결론부터 정하고 사후 합리화"를 가른다.
단, 순서 효과가 항상 정확도를 극적으로 올려주는 건 아니고 케이스/모델에 따라 다르게 나타난다 — 실험해서 확인할 것.
"""
from typing import Literal
from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

SYSTEM = """너는 반도체 회사의 SCM 문의 분류기다.

[category 정의]
- 발주: 구매 요청, 발주 수량/시점 결정, 발주서 작성에 관한 문의
- 재고: 현재 보유량 조회, 창고 현황, 재고 부족 확인에 관한 문의
- 납기: 협력사 입고 지연, 배송 일정, 도착 예정일에 관한 문의
- 기타: SCM과 무관한 문의

[여러 카테고리가 섞인 경우]
문제의 원인에 해당하는 카테고리를 고른다.
예: "주문한 게 안 와서 재고가 없다" → 원인은 입고 지연이므로 납기

[urgency 기준]
- high: 생산 라인 중단 위험, 오늘 안에 조치 필요, 사용자가 긴급함을 명시
- medium: 이번 주 안에 처리하면 되는 업무 문의
- low: 단순 조회, 일반 질문, 업무 무관"""

class ReasonLast(BaseModel):
    # category/urgency를 먼저 결정 → reason은 이미 정해진 답을 사후 정당화하는 문장이 된다.
    # 결론에 영향을 못 주는 "결정 후 합리화" 패턴.
    category: Literal["발주", "재고", "납기", "기타"]
    urgency: Literal["high", "medium", "low"]
    reason: str = Field(description="판단 근거 한 문장")

class ReasonFirst(BaseModel):
    # reason을 먼저 씀 → category/urgency를 정하기 전에 원인을 분석하게 강제된다.
    # pydantic 필드 선언 순서 = model_json_schema()의 properties 순서 = 모델이 토큰을 생성하는 순서.
    # LLM은 autoregressive(왼쪽에서 오른쪽으로, 이미 쓴 토큰을 보며 다음 토큰 생성)라서
    # 이 순서가 곧 "먼저 생각하고 결론 내리기"를 스키마로 강제하는 효과를 낸다.
    reason: str = Field(description="문의의 핵심 원인과 요청 사항을 먼저 분석한 한 문장")
    category: Literal["발주", "재고", "납기", "기타"]
    urgency: Literal["high", "medium", "low"]

def classify(question, schema_cls):
    res = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        temperature=0,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "scm", "schema": schema_cls.model_json_schema()}},
    )
    return schema_cls.model_validate_json(res.choices[0].message.content)

# (문의, 정답 category)
cases = [
    ("납기 일정표 보다가 궁금한 건데, 발주서 양식은 어디 있어요?", "발주"),
    ("협력사가 물량을 절반만 보냈어요. 나머지는 언제 온대요?", "납기"),
    ("라인 멈출 뻔했는데 재고로 버텼어요. 다음엔 미리 넉넉히 주문하려면 어떻게 해요?", "발주"),
    ("재고 부족 알림이 떴는데, 이거 지난주 입고 예정이던 물량이 안 들어와서 그런 거죠?", "납기"),
]

for q, answer in cases:
    print(f"\n문의: {q}  (정답: {answer})")
    for cls in [ReasonLast, ReasonFirst]:
        r = classify(q, cls)
        mark = "✅" if r.category == answer else "❌"
        print(f"  {cls.__name__:11} {mark} {r.category} / {r.urgency} | {r.reason}")