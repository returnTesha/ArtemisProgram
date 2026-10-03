"""
Step 15-1: Self-reflection — 답을 한 번에 확정하지 않고 스스로 비평(critique) → 개선(revise)을 반복.

지금까지(step1~14)의 패턴은 대부분 "한 번 호출하고 그 결과를 바로 최종 답으로 쓴다"였다.
tool 결과처럼 정답이 명확한 사실 조회와 달리, "설명/보고서 작성" 같은 열린 작업은 초안 품질이 들쭉날쭉한데,
자기 자신에게 다시 비평을 시키고(critique) 필요하면 고쳐 쓰게(revise) 하면 빠진 부분을 스스로 채우는 효과가 있다.
draft/critique/revise가 서로 다른 LLM 호출이라는 점에 주목 — "한 번의 똑똑한 호출"이 아니라
"여러 번의 평범한 호출을 역할별로 나눠 조합"해서 품질을 끌어올리는 패턴이다.
"""
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

class Critique(BaseModel):
    has_issues: bool
    feedback: str  # 무엇이 부족한지 (has_issues=False면 빈 문자열)

def draft(question: str) -> str:
    res = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": question}], temperature=0.5)
    return res.choices[0].message.content

def critique(question: str, answer: str) -> Critique:
    prompt = f"""아래는 SCM 관련 질문과 답변이다. 답변에 사실 누락, 과도한 확신, 애매한 표현이 있는지 비평하라.

[질문]
{question}

[답변]
{answer}"""
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_schema", "json_schema": {"name": "critique", "schema": Critique.model_json_schema()}},
    )
    return Critique.model_validate_json(res.choices[0].message.content)

def revise(question: str, answer: str, feedback: str) -> str:
    prompt = f"""[질문]\n{question}\n\n[기존 답변]\n{answer}\n\n[비평]\n{feedback}\n\n비평을 반영해서 답변을 다시 작성하라."""
    res = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
    return res.choices[0].message.content

def answer_with_reflection(question: str, max_rounds: int = 2) -> str:
    answer = draft(question)
    print(f"[초안] {answer}")

    for round_n in range(max_rounds):
        c = critique(question, answer)
        if not c.has_issues:
            # reflection이 항상 "고친다"로 끝나야 하는 건 아니다 — "고칠 게 없다"는 판단도 정상적인 결과.
            print("[비평] 문제 없음 → 종료")
            break
        print(f"[비평 {round_n + 1}] {c.feedback}")
        answer = revise(question, answer, c.feedback)
        print(f"[개선 {round_n + 1}] {answer}")

    return answer

question = "신규 협력사를 선정할 때 어떤 기준으로 평가해야 하는지 SCM 담당자 입장에서 설명해줘"
print(f"질문: {question}\n")
final = answer_with_reflection(question)
print(f"\n최종 답변:\n{final}")
