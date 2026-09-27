"""
Step 2-5: classifier.py를 여러 케이스로 돌려서 "정확도"를 실제로 측정한다.

지금까지는 "이 질문 하나 잘 분류되나"를 눈으로 확인했다면, 여기서는 정답이 정해진 케이스 12개를 모아
category/urgency 각각의 정확도(%)를 집계한다. 100%가 안 나와도 정상 — 작은 로컬 모델의 한계를
숫자로 정직하게 보여주는 게 목적이다 (이런 케이스 세트를 흔히 "평가셋(eval set)"이라 부른다).
"""
from classifier import classify

# (문의, 정답 category, 정답 urgency)
CASES = [
    ("B 협력사 웨이퍼 납기가 3일 늦어진대요. 라인 멈추게 생겼어요", "납기", "high"),
    ("다음 달 포토레지스트 발주 수량 어떻게 정해요?", "발주", "medium"),
    ("창고에 불산 재고 얼마나 남았어?", "재고", "low"),
    ("점심 메뉴 추천해줘", "기타", "low"),
    ("발주 넣은 게 아직 안 왔는데 재고도 바닥이에요. 급해요!!", "납기", "high"),
    ("납기 일정표 보다가 궁금한 건데, 발주서 양식은 어디 있어요?", "발주", "low"),
    ("협력사가 물량을 절반만 보냈어요. 나머지는 언제 온대요?", "납기", "medium"),
    ("라인 멈출 뻔했는데 재고로 버텼어요. 다음엔 미리 넉넉히 주문하려면 어떻게 해요?", "발주", "medium"),
    ("재고 부족 알림이 떴는데, 지난주 입고 예정이던 물량이 안 들어와서 그런 거죠?", "납기", "medium"),
    ("C사 특수가스 오늘 오후까지 안 오면 3라인 세워야 합니다", "납기", "high"),
    ("이번 분기 슬러리 재고 회전율 좀 뽑아줄 수 있어요?", "재고", "low"),
    ("신규 협력사 D사에 처음 발주 넣으려면 절차가 어떻게 돼요?", "발주", "medium"),
]

cat_ok = urg_ok = 0
for q, ans_cat, ans_urg in CASES:
    r = classify(q)
    c, u = r.category == ans_cat, r.urgency == ans_urg
    cat_ok += c
    urg_ok += u
    mark = "✅" if c and u else "❌"
    print(f"{mark} [{r.category}/{r.urgency}] (정답 {ans_cat}/{ans_urg}) item={r.item} supplier={r.supplier}")
    print(f"   {q}")
    if not (c and u):
        print(f"   └ 근거: {r.reason}")

n = len(CASES)
print(f"\ncategory 정확도: {cat_ok}/{n} ({cat_ok / n:.0%})")
print(f"urgency  정확도: {urg_ok}/{n} ({urg_ok / n:.0%})")