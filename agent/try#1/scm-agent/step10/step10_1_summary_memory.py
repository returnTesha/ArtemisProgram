"""
Step 10-1: 요약(summary) 기반 단기 메모리 — step1_4의 "오래된 대화 통째로 삭제"를 개선한 버전.

step1_4_chatbot.py는 토큰이 넘치면 가장 오래된 turn을 그냥 잘라서 버렸다 (정보 손실).
여기서는 버리는 대신, 그 오래된 대화를 LLM에게 다시 요약시켜서 짧은 문자열 하나로 압축해 남긴다.
그래서 "아까 말한 협력사 이름"처럼 이미 잘려나갔을 정보도, 요약 안에 핵심만 남아있으면 여전히 답할 수 있다.
"반드시 한국어로만"을 프롬프트에 박아둔 이유: qwen2.5:7b가 요약 작업에서 가끔 중국어를 섞어 쓰는 걸
실제로 관찰해서 고친 것 — 작은 모델일수록 출력 언어를 명시적으로 고정해야 한다 (README 참고).
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

def summarize(history: list[dict]) -> str:
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history if m["role"] != "system")
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"다음 대화를 이후 대화에서 참고할 수 있도록 핵심만 3줄 이내로, 반드시 한국어로만 요약해줘:\n\n{convo}"}],
        temperature=0,
    )
    return res.choices[0].message.content

class SummaryMemoryChat:
    """전체 히스토리를 다 보내는 대신, 오래된 대화는 요약 하나로 압축해서 토큰을 아낀다."""

    def __init__(self, system: str, keep_recent: int = 2):
        self.system = {"role": "system", "content": system}
        self.summary = None
        self.recent: list[dict] = []  # 최근 turn은 그대로 보존 (요약하면 디테일이 날아가므로)
        self.keep_recent = keep_recent

    def _build_messages(self):
        # 매 호출마다 [system, (요약이 있으면) 요약 system 메시지, 최근 대화들]을 다시 조립해서 보낸다.
        # step1_3의 stateless 원칙 그대로 — 서버는 아무것도 기억하지 않으므로 압축된 컨텍스트를 매번 재구성해야 한다.
        messages = [self.system]
        if self.summary:
            messages.append({"role": "system", "content": f"[이전 대화 요약]\n{self.summary}"})
        messages.extend(self.recent)
        return messages

    def ask(self, user_input: str) -> str:
        self.recent.append({"role": "user", "content": user_input})
        res = client.chat.completions.create(model=MODEL, messages=self._build_messages(), temperature=0.3)
        answer = res.choices[0].message.content
        self.recent.append({"role": "assistant", "content": answer})

        # turn 수가 넘치면 오래된 것들을 요약에 흡수시키고 recent에서 제거
        if len(self.recent) > self.keep_recent * 2:
            to_summarize = self.recent[: -self.keep_recent * 2] if self.keep_recent else self.recent
            old_summary_input = ([{"role": "system", "content": f"[기존 요약] {self.summary}"}] if self.summary else []) + to_summarize
            self.summary = summarize(old_summary_input)
            self.recent = self.recent[-self.keep_recent * 2:]
            print(f"  [메모리 압축됨] 요약: {self.summary}")

        return answer

chat = SummaryMemoryChat(system="너는 SCM 도우미다. 한국어로 간결하게 답한다.", keep_recent=1)

for q in [
    "B 협력사 웨이퍼 납기가 3일 지연됐어",
    "그럼 대체 협력사를 알아봐야 하나?",
    "혹시 아까 말한 협력사 이름이 뭐였지?",
]:
    print(f"\n나: {q}")
    print("도우미:", chat.ask(q))
