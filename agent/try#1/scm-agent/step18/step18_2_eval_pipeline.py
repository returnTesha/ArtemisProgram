"""
Step 18-2: step18_1과 같은 LLM-as-judge 방식(judge()는 이 파일에 프롬프트를 줄여 다시 정의함)으로,
step7의 RAG 파이프라인 전체를 평가한다.

분류 정확도(step2_5)처럼 개별 컴포넌트가 아니라, "검색 + 생성"이 합쳐진 최종 답변 품질을 평가하는 것.
RAG 파이프라인은 검색이 틀려도, 생성이 틀려도 최종 답이 틀리게 나오므로, 이 evaluation은 두 문제를
구분하지 않고 "최종 사용자가 받는 답이 맞았는가"만 본다 — 그래서 통과율이 낮으면 검색/생성 중
어디가 문제인지는 이 결과만으론 알 수 없고, 단계별로 더 파고들어야 한다.
"""
import sys, os
from typing import Literal
from openai import OpenAI
from pydantic import BaseModel, Field
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "step7"))
from step7_3_rag_pipeline import rag_answer

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

class Judgment(BaseModel):
    reasoning: str = Field(description="판정 근거를 먼저 한 문장으로 설명")
    verdict: Literal["pass", "fail"]

def judge(question: str, reference: str, answer: str) -> Judgment:
    prompt = f"""[기준 답변]의 핵심 내용을 [실제 답변]이 정확히 포함하면 pass, 아니면 fail로 판정하라.

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

# (질문, 기준 답변) — RAG 파이프라인 전체(검색+생성)를 통째로 평가한다.
EVAL_SET = [
    ("긴급 발주는 팀장 승인 없이도 가능해?", "구매팀장 직권으로 가능하지만, 24시간 이내 사후 결재가 필요하다."),
    ("협력사 A등급 기준이 뭐야?", "납기 준수율 90% 이상이면 A등급이다."),
    ("납기가 며칠 지연되면 대체 협력사를 검토해?", "3일을 넘으면 대체 협력사 소싱을 검토해야 한다."),
    ("회사 주가는 얼마야?", "규정에서 확인할 수 없다는 취지로 답해야 한다."),
    ("긴급 발주는 팀장 승인 없이도 가능해?", "불가능합니다"),
]

results = []
for question, reference in EVAL_SET:
    answer = rag_answer(question)
    j = judge(question, reference, answer)
    results.append((question, reference, answer, j))

passed = sum(1 for *_ , j in results if j.verdict == "pass")
for question, reference, answer, j in results:
    mark = "✅" if j.verdict == "pass" else "❌"
    print(f"{mark} 질문: {question}")
    print(f"   기준: {reference}")
    print(f"   실제: {answer}")
    if j.verdict == "fail":
        print(f"   실패 사유: {j.reasoning}")
    print()

print(f"통과율: {passed}/{len(EVAL_SET)} ({passed / len(EVAL_SET):.0%})")
