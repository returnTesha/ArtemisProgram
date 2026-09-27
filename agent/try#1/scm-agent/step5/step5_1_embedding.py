"""
Step 5-1: 텍스트를 벡터로 바꾸는 임베딩 API의 가장 기본 형태.

chat completions(step1~4)와 결정적으로 다른 점: embeddings API엔 messages/role 구조가 없다.
system 프롬프트를 넣을 자리 자체가 없다는 뜻 — 그래서 이후 step5_2/5_3에서
"임베딩 결과는 프롬프트로 못 고친다"는 한계가 드러난다.

또한 이 변환은 학습 시점에 이미 고정된(frozen) 가중치를 그대로 적용하는 결정론적 계산이라,
같은 텍스트를 넣으면 몇 번을 호출하든 항상 완전히 동일한 벡터가 나온다 (chat과 달리 temperature도 없음).
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

res = client.embeddings.create(model="bge-m3", input="재고가 부족해요")
vector = res.data[0].embedding

print("벡터 차원 수:", len(vector))  # bge-m3는 1024차원 — 이 숫자가 이후 RAG 스텝에서 계속 등장한다
print("앞부분 5개 값:", vector[:5])  # 숫자 자체는 사람이 읽어서 의미를 알 수 없다. 벡터 간 "거리"만 의미가 있다 (step5_2)
