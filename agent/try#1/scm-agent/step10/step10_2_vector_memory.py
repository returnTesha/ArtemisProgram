"""
Step 10-2: 벡터(임베딩) 기반 장기 메모리 — step5의 코사인 유사도 검색을 "기록 저장소"에 적용한 것.

step9의 search_policy(고정된 사내 문서 검색)와 구조는 완전히 같다. 차이는 검색 대상이
"미리 준비된 문서"가 아니라 "과거에 쌓인 대화/이벤트 기록"이라는 점 — 즉 메모리도 결국 미니 RAG다.
recall()의 threshold 인자(기본 0.5)가 핵심 안전장치: 유사도가 일정 기준 밑이면 아예 아무것도 안 꺼내온다.
이게 없으면 "오늘 날씨 어때?"처럼 관련 기록이 전혀 없는 질문에도 가장 비슷한(사실은 안 비슷한) 기록을
억지로 끌어와서 답변을 오염시키게 된다.
"""
import numpy as np
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"

def embed(text: str) -> np.ndarray:
    return np.array(client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class VectorMemory:
    """요약 메모리는 '최근' 맥락을 압축한다. 벡터 메모리는 '오래전' 기록이라도
    지금 질문과 관련 있으면 다시 꺼내오는 검색 방식을 보여준다.
    주의: 저장소는 프로세스 메모리 안의 리스트(self.entries)일 뿐이라 프로그램이 끝나면 사라진다.
    실제 세션을 넘는 영속성은 디스크/벡터 DB에 저장해야 가능하다 (step7 참고)."""

    def __init__(self):
        self.entries: list[tuple[str, np.ndarray]] = []  # (기록 텍스트, 임베딩)

    def remember(self, text: str):
        self.entries.append((text, embed(text)))

    def recall(self, query: str, top_k: int = 2, threshold: float = 0.5) -> list[str]:
        if not self.entries:
            return []
        query_vec = embed(query)
        scored = sorted(((cosine_similarity(query_vec, v), t) for t, v in self.entries), reverse=True)
        return [t for score, t in scored[:top_k] if score >= threshold]

memory = VectorMemory()

# 지난주, 지난달 등 완전히 다른 세션에서 있었던 일들이라고 가정 (실제로는 실행할 때마다 이 remember() 호출로 다시 채운다)
memory.remember("2026-09-10: A 협력사 특수가스 납기가 5일 지연되어 3라인이 2시간 멈췄음. 원인은 A사 통관 지연.")
memory.remember("2026-09-15: 포토레지스트 발주량을 기존 대비 20% 늘리기로 결정함 (분기 수요 증가 예측).")
memory.remember("2026-09-18: 신규 협력사 D사 품질 샘플 검사 통과, 다음 분기부터 거래 시작 예정.")

def ask_with_memory(question: str) -> tuple[list[str], str]:
    recalled = memory.recall(question)
    context = "\n".join(f"- {r}" for r in recalled)
    system = "너는 SCM 도우미다. 아래 [과거 기록]을 참고해서 한국어로 답하라. 관련 기록이 없으면 참고하지 말고 그냥 답하라."
    if context:
        system += f"\n\n[과거 기록]\n{context}"

    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
        temperature=0,
    )
    return recalled, res.choices[0].message.content

for q in ["예전에 라인이 멈춘 적 있었나?", "포토레지스트 발주량 관련해서 최근에 뭔가 정해진 게 있어?", "오늘 날씨 어때?"]:
    recalled, answer = ask_with_memory(q)
    print(f"\n질문: {q}")
    print("떠올린 기록:", recalled if recalled else "(없음)")
    print("답변:", answer)
