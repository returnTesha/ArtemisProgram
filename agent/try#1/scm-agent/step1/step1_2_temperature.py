"""
Step 1-2: temperature(온도)가 답변의 일관성에 미치는 영향.

같은 질문을 온도만 바꿔서 3번씩 반복 호출한다.
temperature는 다음 토큰을 고를 때 얼마나 "무작위성"을 허용할지를 정하는 값이다.
- 0: 항상 확률이 가장 높은 토큰만 선택 → 사실상 결정론적(같은 입력엔 거의 항상 같은 출력)
- 높을수록: 확률이 낮은 토큰도 종종 선택 → 매번 표현이 달라짐, 창의적이지만 일관성은 떨어짐
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

for temp in [0, 1.5]:
    print(f"\n=== temperature={temp} ===")
    for i in range(3):
        res = client.chat.completions.create(
            model="qwen2.5:7b",
            messages=[
                {"role": "system", "content": "너는 한 문장으로만 답한다."},
                {"role": "user", "content": "REST API가 뭐야?"},
            ],
            temperature=temp,
        )
        print(f"{i + 1}. {res.choices[0].message.content}")

# 관찰 포인트: temperature=0일 때 3번 답이 (거의) 똑같이 나오고,
# temperature=1.5일 때는 표현이 매번 크게 달라지는지 비교해볼 것.
# 분류/추출처럼 "정답이 하나"인 작업은 temperature=0을 쓰는 이유가 여기서 나온다 (step2에서 실제로 그렇게 함).
