"""
Step 5-2: 임베딩 벡터로 "의미가 비슷한 문장 찾기"를 구현한다.

코사인 유사도(cos θ = A·B / |A||B|)를 쓰는 이유는 임의 선택이 아니다:
- 벡터 내적(dot product)의 정의 자체에서 바로 나오는 값이라, 각도를 따로 구할 필요 없이
  A·B와 |A|,|B|만 계산하면 됨 → 몇 차원이든(1024차원이든 수만 차원이든) 그대로 적용 가능.
- 반대로 sin은 방향이 같을 때(0°) 0이 나와 유사도 지표로 방향이 뒤집히고,
  0°와 180°에서 값이 같아 "동일"과 "정반대"를 구분 못 한다. tan은 90°에서 발산한다.
- 게다가 sin/tan을 구하려면 외적(cross product)이 필요한데, 외적은 3차원에서만 깔끔히 정의된다 —
  임베딩처럼 고차원 벡터에선 애초에 계산할 방법이 없다.
"""
import numpy as np
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embed(text: str):
    res = client.embeddings.create(model="bge-m3", input=text)
    return res.data[0].embedding

query = "창고에 재고가 없어요"

candidates = [
    "재고가 부족해요",              # 단어는 다르지만 의미가 거의 동일 → 유사도가 가장 높아야 함
    "B 협력사 납기가 지연됐어요",    # SCM 도메인상 재고 부족의 흔한 "원인" — 그런데 bge-m3는 이 인과관계를 모른다
    "오늘 점심 메뉴 추천해줘",       # 완전히 무관
    "입고 예정일이 궁금해요",
]

query_vec = embed(query)  # 질문은 한 번만 임베딩해서 재사용 (매번 다시 계산할 필요 없음)

print(f"질문: {query}\n")
for c in candidates:
    sim = cosine_similarity(query_vec, embed(c))
    print(f"{sim:.4f}  {c}")

# 관찰 포인트: "B 협력사 납기가 지연됐어요"와 "오늘 점심 메뉴 추천해줘"의 점수 차이가
# 생각보다 크지 않다 (실측 시 0.05 안팎) — 임베딩은 표면적 단어 유사성만 볼 뿐,
# "납기 지연 → 재고 부족"이라는 도메인 인과관계는 전혀 반영하지 못한다는 한계를 보여준다.
# 이 한계를 system 프롬프트로 못 고치는 이유(embeddings API엔 프롬프트를 넣을 자리가 없음)와
# 그 대안(LLM 재순위화)은 step5_3_rerank.py에서 다룬다.
