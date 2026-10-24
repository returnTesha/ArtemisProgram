"""
Step 19-1: Observability(관찰가능성) — 여러 단계짜리 에이전트에서 "어느 단계가 틀렸는지" 자동 기록.

에이전트가 분류 → 검색 → 도구 호출 → 생성처럼 여러 단계를 거치면, 최종 답이 이상해도
그게 어느 단계의 문제인지 눈으로는 알기 어렵다. @traced 데코레이터로 함수를 감싸면
"언제 시작했는지, 뭘 받았는지, 뭘 반환했는지, 얼마나 걸렸는지"가 TRACE_LOG에 자동으로 쌓인다 —
비즈니스 로직(check_stock, llm_generate) 코드는 한 줄도 안 건드리고 관찰 기능만 덧씌운 것.
"""
import functools
import time
import uuid
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

TRACE_LOG: list[dict] = []

def traced(name: str):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            span_id = str(uuid.uuid4())[:8]
            start = time.time()
            print(f"  → [{span_id}] {name} 시작 args={args} kwargs={kwargs}")
            try:
                result = fn(*args, **kwargs)
                TRACE_LOG.append({
                    "span_id": span_id, "name": name, "duration_ms": round((time.time() - start) * 1000, 1),
                    "input": {"args": args, "kwargs": kwargs}, "output": result, "error": None,
                })
                print(f"  ← [{span_id}] {name} 완료 ({TRACE_LOG[-1]['duration_ms']}ms)")
                return result
            except Exception as e:
                TRACE_LOG.append({
                    "span_id": span_id, "name": name, "duration_ms": round((time.time() - start) * 1000, 1),
                    "input": {"args": args, "kwargs": kwargs}, "output": None, "error": str(e),
                })
                print(f"  ✗ [{span_id}] {name} 실패: {e}")
                raise
        return wrapper
    return decorator

FAKE_STOCK_DB = {"웨이퍼": 1200, "포토레지스트": 340, "불산": 15}

@traced("check_stock")
def check_stock(item: str) -> dict:
    qty = FAKE_STOCK_DB.get(item)
    return {"item": item, "quantity": qty} if qty is not None else {"item": item, "found": False}

@traced("llm_generate")
def llm_generate(prompt: str) -> str:
    res = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
    return res.choices[0].message.content

def run_agent(question: str) -> str:
    stock = check_stock("웨이퍼")
    return llm_generate(f"질문: {question}\n재고 정보: {stock}\n한국어로 답하라.")

answer = run_agent("웨이퍼 재고 알려줘")
print("\n최종 답변:", answer)

print("\n=== 추적 로그(trace) 전체 ===")
for span in TRACE_LOG:
    print(f"[{span['span_id']}] {span['name']} — {span['duration_ms']}ms — 입력={span['input']['args']} → 출력={str(span['output'])[:60]}")
