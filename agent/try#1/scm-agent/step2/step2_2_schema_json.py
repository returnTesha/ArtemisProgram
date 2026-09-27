"""
Step 2-2: response_format(JSON Schema)으로 출력 형식을 "강제"한다.

step2_1과 똑같은 프롬프트/질문 세트인데, 이번엔 SCHEMA를 response_format에 넘겨서
서버가 아예 스키마를 벗어난 JSON을 생성 못 하게 제약한다 (enum으로 category/urgency 값도 제한).
그 결과 step2_1에서 가끔 나오던 파싱 실패나 "발주|재고|납기|기타" 같은 짬뽕 값이 사라지는지 확인하는 게 관찰 포인트.
"""
import json
from openai import OpenAI

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

SCHEMA = {
    "type": "object",
    "properties": {
        # enum으로 허용 값을 못박아버리면, 모델이 "발주|재고|납기|기타" 같은 값을 아예 생성할 수 없게 된다.
        "category": {"type": "string", "enum": ["발주", "재고", "납기", "기타"]},
        "urgency": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["category", "urgency"],
}

questions = [
    "B 협력사 웨이퍼 납기가 3일 늦어진대요. 라인 멈추게 생겼어요",
    "다음 달 포토레지스트 발주 수량 어떻게 정해요?",
    "창고에 불산 재고 얼마나 남았어?",
    "점심 메뉴 추천해줘",
    "발주 넣은 게 아직 안 왔는데 재고도 바닥이에요. 급해요!!",
]

for q in questions:
    res = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "scm_classify", "schema": SCHEMA},
        },
    )
    data = json.loads(res.choices[0].message.content)
    print(f"{data['category']:4} | {data['urgency']:6} | {q}")