"""
step2 시리즈의 최종 결과물: 재사용 가능한 분류 함수 하나로 정리.

step2_1(프롬프트만) → step2_2(JSON Schema 강제) → step2_3(Pydantic 검증) → step2_4(필드 순서 실험)
까지 배운 걸 다 합쳐서, 다른 스크립트가 `from classifier import classify`로 바로 가져다 쓸 수 있게 만든 것.
step2_5의 정확도 평가, 이후 step3의 tool calling 등에서 이 함수를 그대로 재사용한다.
"""
from typing import Literal, Optional
from openai import OpenAI
from pydantic import BaseModel, Field
from prompts import CLASSIFY_SYSTEM

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

class Classification(BaseModel):
    # reason을 맨 앞에 둔 이유: step2_4에서 확인했듯, 구조화 출력의 필드 순서 = 모델이 토큰을 생성하는 순서다.
    # LLM은 autoregressive(이미 뱉은 토큰을 보며 다음 토큰 생성)라서, reason을 먼저 쓰게 하면
    # "먼저 생각하고 결론 내리기"가 되고, 뒤에 두면 "결론부터 정하고 사후 합리화"가 된다.
    reason: str = Field(description="문의의 핵심 원인과 요청 사항을 먼저 분석한 한 문장")
    category: Literal["발주", "재고", "납기", "기타"]
    urgency: Literal["high", "medium", "low"]
    item: Optional[str] = Field(default=None, description="문의에 언급된 품목명. 없으면 null")
    supplier: Optional[str] = Field(default=None, description="문의에 언급된 협력사명. 없으면 null")

def classify(question: str) -> Classification:
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": CLASSIFY_SYSTEM},
                  {"role": "user", "content": question}],
        temperature=0,  # 분류처럼 "정답이 하나"인 작업은 무작위성을 끈다 (step1_2에서 확인한 원칙)
        response_format={"type": "json_schema",
                         # Pydantic 모델에서 스키마를 자동 생성 → 필드 타입(Literal 등)이 곧 enum 제약이 된다
                         "json_schema": {"name": "scm_classify",
                                         "schema": Classification.model_json_schema()}},
    )
    # model_validate_json: 문자열로 온 JSON을 Classification으로 검증까지 하며 파싱.
    # category/urgency가 Literal 범위를 벗어나면 여기서 ValidationError가 나서 잘못된 값을 코드가 확실히 잡아낸다.
    return Classification.model_validate_json(res.choices[0].message.content)

if __name__ == "__main__":
    r = classify("B 협력사 웨이퍼 납기가 3일 늦어진대요. 라인 멈추게 생겼어요")
    print(r.model_dump_json(indent=2))