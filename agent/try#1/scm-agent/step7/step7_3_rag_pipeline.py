"""
Step 7-3: step6의 순수 코드 RAG를 ChromaDB 기반으로 다시 구현 — retrieve만 바뀌고 구조는 동일하다.

retrieve()가 "파이썬 리스트에서 코사인 유사도 계산"(step6_2)에서 "ChromaDB collection.query()"로
바뀐 것 말고는, augment(프롬프트 조립)와 generate(LLM 호출) 단계는 개념적으로 완전히 같다.
이걸 통해 "RAG = Retrieve → Augment → Generate"라는 구조 자체는 벡터 저장 방식과 무관하게
그대로 유지된다는 걸 확인할 수 있다 — 벡터DB는 R 단계를 더 잘 하게 해주는 도구일 뿐이다.
"""
import os
import chromadb
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
CHAT_MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"

CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_store")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name="scm_docs")

def embed(text: str) -> list[float]:
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def retrieve(query: str, top_k: int = 3) -> list[str]:
    result = collection.query(query_embeddings=[embed(query)], n_results=top_k)
    return result["documents"][0]

def augment(question: str, context_docs: list[str]) -> str:
    context = "\n".join(f"- {d}" for d in context_docs)
    return f"""아래 [사내 규정]만 근거로 질문에 답하라. 규정에 없는 내용은 "규정에서 확인할 수 없습니다"라고 답하라.

[사내 규정]
{context}

[질문]
{question}"""

def generate(prompt: str) -> str:
    res = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
    return res.choices[0].message.content

def rag_answer(question: str) -> str:
    # RAG = Retrieve → Augment → Generate, 이 세 단계가 전부다.
    docs = retrieve(question)
    prompt = augment(question, docs)
    return generate(prompt)

if __name__ == "__main__":
    for q in ["신규 협력사 등록하려면 뭘 통과해야 해?", "위험물 검수는 혼자 해도 돼?"]:
        print(f"\n질문: {q}")
        print("답변:", rag_answer(q))
