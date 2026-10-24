"""
Step 19-2: Metrics — 요청 하나하나가 아니라 "전체적으로" 얼마나 느리고 비싼지 집계한다.

step19_1의 tracing이 "요청 하나가 어떻게 처리됐는지"를 보는 것이라면, 이 파일의 metrics는
여러 요청을 모아서 지연시간(p50/p95)과 토큰 사용량을 숫자로 요약한다.
평균만 보면 "가끔 아주 느린 요청"이 묻히기 때문에 percentile을 따로 뽑는다.
로컬 모델은 토큰이 공짜지만, cost_per_1m_* 파라미터는 유료 API(GPT/Claude 등)로 바꾸는 순간
그대로 실제 비용 추정치가 된다.
"""
import time
from dataclasses import dataclass, field
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

@dataclass
class Metrics:
    latencies_ms: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def record(self, latency_ms: float, prompt_tokens: int, completion_tokens: int):
        self.latencies_ms.append(latency_ms)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def percentile(self, p: float) -> float:
        # p95 = "95%의 요청이 이 시간 안에 끝난다" — 정렬 후 그 위치의 값을 그냥 읽으면 된다 (선형보간 없는 간단 버전)
        s = sorted(self.latencies_ms)
        idx = min(int(len(s) * p), len(s) - 1)
        return round(s[idx], 1)

    def summary(self, cost_per_1m_input: float = 0.15, cost_per_1m_output: float = 0.60) -> dict:
        total_tokens = self.prompt_tokens + self.completion_tokens
        estimated_cost = (self.prompt_tokens / 1_000_000 * cost_per_1m_input) + (self.completion_tokens / 1_000_000 * cost_per_1m_output)
        return {
            "요청 수": len(self.latencies_ms),
            "평균 지연(ms)": round(sum(self.latencies_ms) / len(self.latencies_ms), 1),
            "p50 지연(ms)": self.percentile(0.5),
            "p95 지연(ms)": self.percentile(0.95),
            "총 토큰": total_tokens,
            "예상 비용(USD, 유료 API 기준 가정치)": round(estimated_cost, 6),
        }

metrics = Metrics()

def call_and_record(question: str) -> str:
    start = time.time()
    res = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": question}], temperature=0)
    latency_ms = (time.time() - start) * 1000
    metrics.record(latency_ms, res.usage.prompt_tokens, res.usage.completion_tokens)
    return res.choices[0].message.content

questions = [
    "발주 절차를 한 문장으로 설명해줘",
    "재고 관리가 왜 중요한지 한 문장으로 설명해줘",
    "납기 지연이 생기면 뭘 해야 하는지 한 문장으로 설명해줘",
]

for q in questions:
    answer = call_and_record(q)
    print(f"질문: {q}\n답변: {answer}\n")

print("=== 집계 메트릭 ===")
for k, v in metrics.summary().items():
    print(f"{k}: {v}")
