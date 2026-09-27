"""
Step 2-3: JSON Schema(step2_2) 위에 Pydantic 검증까지 한 겹 더 얹는다.

response_format으로 형식을 강제하는 건 "서버(모델)가 스키마를 지키게 만드는" 것이고,
Pydantic으로 검증하는 건 "그 결과를 우리 코드가 신뢰하기 전에 한 번 더 확인하는"것 — 서로 다른 안전장치다.
Literal[...] 타입은 pydantic이 자동으로 model_json_schema()에 enum으로 반영해주기도 하고,
동시에 model_validate_json()에서 실제 검증기로도 동작한다 (1석 2조).
"""
from typing import Literal
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

class Classification(BaseModel):
    category: Literal["발주", "재고", "납기", "기타"]
    urgency: Literal["high", "medium", "low"]
    reason: str = Field(description="판단 근거를 한 문장으로")

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

def classify(question: str) -> Classification:
    res = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "scm_classify", "schema": Classification.model_json_schema()},
        },
    )
    return Classification.model_validate_json(res.choices[0].message.content)

# 1) 정상 동작
# 여러 카테고리가 섞인 애매한 질문 (step2_1에서 "발주|재고|납기|기타"로 뭉개졌던 바로 그 질문).
# SYSTEM 프롬프트의 "원인에 해당하는 카테고리를 고른다" 규칙 덕분에 category='납기' 하나로 정확히 나온다.
result = classify("발주 넣은 게 아직 안 왔는데 재고도 바닥이에요. 급해요!!")
print(result)
print(f"분기 처리: {result.category} 담당자에게 전달 (긴급도 {result.urgency})")
print(f"근거: {result.reason}")

# 2) 검증 실패 시뮬레이션: 실제 모델 응답이 아니라 하드코딩한 문자열이다.
# "스키마 강제 없이 모델이 step2_1처럼 잘못된 값을 뱉었다면" 이라는 가정 하에,
# Pydantic의 Literal 검증이 그 값을 확실히 걸러내는지 보여주기 위한 것.
bad = '{"category": "발주|재고|납기|기타", "urgency": "high", "reason": "복합 문의"}'
try:
    Classification.model_validate_json(bad)
except ValidationError as e:
    print("\n검증 실패로 잡아냄:")
    print(e)