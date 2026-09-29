"""
Step 6-1: RAG의 첫 단계인 "chunking(문서 쪼개기)"이 왜 결과 품질을 좌우하는지 확인한다.

RAG는 질문이 오면 미리 쪼개둔 chunk 중 관련 있는 걸 찾아 모델에 넘긴다 (retrieve).
그런데 chunk를 어떻게 나누느냐에 따라 "관련 있는 내용이 통째로 한 chunk에 담기는지",
아니면 "문장 중간이 잘려서 앞뒤 맥락이 흩어지는지"가 갈린다 — 검색 품질의 출발점.
"""
from docs import SCM_DOCS

RAW_TEXT = " ".join(SCM_DOCS)

def chunk_by_length(text: str, size: int = 80, overlap: int = 20) -> list[str]:
    # 글자 수로 기계적으로 자름 → 문장이 어디서 끝나는지 신경 안 쓰기 때문에 중간에 뚝뚝 끊긴다.
    # overlap을 두는 이유: 안 겹치게 자르면 경계에 걸친 내용이 어느 chunk에도 온전히 안 담길 수 있어서다.
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def chunk_by_sentence(docs: list[str]) -> list[str]:
    # 문서 하나 = 이미 하나의 완결된 규정 문장이므로, 여기서는 문서 단위가 곧 chunk 단위다.
    return docs

print("=== 방식 A: 글자 수로 자르기 (문맥이 중간에 끊길 수 있음) ===")
for i, c in enumerate(chunk_by_length(RAW_TEXT)[:3]):
    print(f"[{i}] {c}")

print("\n=== 방식 B: 문서(문장) 단위로 자르기 (권장) ===")
for i, c in enumerate(chunk_by_sentence(SCM_DOCS)[:3]):
    print(f"[{i}] {c}")

print(f"\n총 문서 수: {len(SCM_DOCS)}개, 방식 A로는 {len(chunk_by_length(RAW_TEXT))}개 chunk 생성")
print("→ 글자 수로 자르면 문장 중간이 잘려 검색 정확도가 떨어진다. 실무에서는 의미 단위(문단/문장)로 자르는 게 기본이다.")
