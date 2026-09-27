"""
Step 2-1: "JSON으로만 답해"라는 프롬프트 지시만으로 구조화 출력을 시도.

여기서는 JSON Schema 강제(response_format)를 아직 쓰지 않고, 오직 system 프롬프트 문구로만 형식을 요청한다.
이 방식의 한계를 두 겹으로 확인하는 게 목적:
  1) json.loads()가 실패하는 경우 (모델이 JSON 문법 자체를 어길 때)
  2) json.loads()는 성공하지만 값이 이상한 경우 — 예: category에 "발주|재고|납기|기타"처럼
     여러 카테고리가 섞인 애매한 질문에서 모델이 하나를 못 고르고 프롬프트의 예시 문구를 그대로 베낌.
     "파싱 성공"과 "값이 맞음"은 다른 검증이라는 걸 보여주려고 CATEGORIES/URGENCIES 체크를 따로 둔다.
"""
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

CATEGORIES = ["발주", "재고", "납기", "기타"]
URGENCIES = ["high", "medium", "low"]

SYSTEM = """너는 SCM 문의 분류기다.
사용자 문의를 읽고 아래 JSON 형식으로만 답하라. 다른 텍스트는 절대 포함하지 마라.
{"category": "<분류>", "urgency": "<긴급도>"}

<분류>는 반드시 다음 4개 중 하나만 골라라: 발주, 재고, 납기, 기타
- 문의가 여러 주제에 걸쳐 있어도 가장 시급하거나 핵심적인 주제 하나만 선택한다.
- 카테고리를 "|"로 나열하거나 여러 개를 동시에 쓰지 않는다.

<긴급도>는 반드시 다음 3개 중 하나만 골라라: high, medium, low

예시:
문의: 발주 넣은 게 아직 안 왔는데 재고도 바닥이에요. 급해요!!
답: {"category": "재고", "urgency": "high"}"""

questions = [
    "B 협력사 웨이퍼 납기가 3일 늦어진대요. 라인 멈추게 생겼어요",
    "다음 달 포토레지스트 발주 수량 어떻게 정해요?",
    "창고에 불산 재고 얼마나 남았어?",
    "점심 메뉴 추천해줘",
    "발주 넣은 게 아직 안 왔는데 재고도 바닥이에요. 급해요!!",
]

success = 0
for q in questions:
    res = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
        temperature=0,
    )
    raw = res.choices[0].message.content
    print(f"\n문의: {q}")
    print(f"원본: {raw}")
    try:
        data = json.loads(raw)
        category, urgency = data["category"], data["urgency"]
        if category not in CATEGORIES:
            print(f"검증 실패 → category 값이 유효하지 않음: {category!r}")
        elif urgency not in URGENCIES:
            print(f"검증 실패 → urgency 값이 유효하지 않음: {urgency!r}")
        else:
            print(f"파싱 성공 → category={category}, urgency={urgency}")
            success += 1
    except Exception as e:
        print(f"파싱 실패 → {type(e).__name__}: {e}")

print(f"\n결과: {success}/{len(questions)} 성공")