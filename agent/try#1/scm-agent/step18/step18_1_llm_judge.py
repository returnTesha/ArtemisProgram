"""
Step 18-1: LLM-as-judge — 정답이 하나로 정해지지 않는 자유 서술형 답변을 자동 채점한다.

step2의 step2_5_eval_classify.py는 "정답이 정해진" 분류를 정확도(%)로 채점했다.
하지만 자유 서술형 답변은 표현이 달라도 핵심 사실만 맞으면 정답이다 — 문자열을 그대로 비교할 수 없다.
이럴 때 쓰는 게 LLM-as-judge: 채점 기준을 프롬프트에 명시하고, LLM에게 "심사위원" 역할을 시킨다.
주의: 심사하는 LLM도 틀릴 수 있으므로, 중요한 결정에는 사람 검토를 곁들여야 한다.
"""
from typing import Literal
from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

class Judgment(BaseModel):
    reasoning: str = Field(description="핵심 사실이 맞는지 틀린지 먼저 한 문장으로 분석")
    verdict: Literal["pass", "fail"]

def judge(question: str, reference: str, answer: str) -> Judgment:
    prompt = f"""너는 SCM 도우미의 답변을 채점하는 심사위원이다. 아래 [기준 답변]의 핵심 내용을 [실제 답변]이 빠짐없이, 정확하게 포함하는지 평가하라.
표현이 다르더라도 핵심 사실이 맞으면 pass, 핵심 사실이 틀리거나 빠지면 fail이다.

[질문]
{question}

[기준 답변]
{reference}

[실제 답변]
{answer}"""
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_schema", "json_schema": {"name": "judgment", "schema": Judgment.model_json_schema()}},
    )
    return Judgment.model_validate_json(res.choices[0].message.content)

# 처음엔 reasoning + verdict(pass/fail) + score(1~5) 세 필드로 만들었는데,
# 작은 로컬 모델은 "reasoning은 맞다고 해놓고 score는 1점" 같은 자기모순을 자주 냈다.
# 필드가 늘어날수록 서로 논리적으로 어긋날 여지가 커지므로, score를 없애고
# reasoning + verdict 두 필드만 남겼다 (reasoning을 verdict보다 먼저 두어 "분석 후 판정" 순서로 생성시킴).
# 세분화된 점수가 필요하면 verdict별로 사람이 별도 기준을 정하는 게 낫다.

cases = [
    ("웨이퍼 재고 얼마나 남았어?", "웨이퍼 재고는 1200kg 남아있다.", "현재 웨이퍼는 1200kg 보유하고 있습니다."),
    ("웨이퍼 재고 얼마나 남았어?", "웨이퍼 재고는 1200kg 남아있다.", "웨이퍼 재고는 500kg입니다."),
    ("B 협력사 납기 지연됐어?", "3일 지연됐다.", "네, 지연되긴 했는데 정확한 기간은 모르겠습니다."),
]

for question, reference, answer in cases:
    j = judge(question, reference, answer)
    mark = "✅" if j.verdict == "pass" else "❌"
    print(f"{mark} 답변: {answer}")
    print(f"   근거: {j.reasoning}\n")
