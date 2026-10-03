"""
Step 13-1: 미니 그래프 오케스트레이션 엔진 직접 구현.

"그래프 AI"(LangGraph 등)의 핵심 아이디어는 단순하다:
- 상태(state)는 딕셔너리 하나로 노드들 사이를 흘러다닌다.
- 노드(node)는 상태를 받아서 상태를 갱신하는 함수다.
- 엣지(edge)는 "이 노드 다음엔 저 노드"를 정하고, 조건부 엣지는 상태를 보고 분기한다.
이것만 구현하면 LangGraph가 하는 일의 본질을 이해할 수 있다.
이 Graph 클래스는 그대로 step13_2(SCM 워크플로), 그리고 step20/step30 캡스톤 에이전트에서도 재사용된다.
"""
from typing import Callable

END = "__end__"

class Graph:
    def __init__(self):
        self.nodes: dict[str, Callable[[dict], dict]] = {}
        self.edges: dict[str, str] = {}
        self.conditional_edges: dict[str, Callable[[dict], str]] = {}
        self.entry: str | None = None

    def add_node(self, name: str, fn: Callable[[dict], dict]):
        self.nodes[name] = fn
        return self

    def set_entry(self, name: str):
        self.entry = name
        return self

    def add_edge(self, from_node: str, to_node: str):
        self.edges[from_node] = to_node
        return self

    def add_conditional_edge(self, from_node: str, router: Callable[[dict], str]):
        # router(state) -> 다음 노드 이름 (또는 END)
        self.conditional_edges[from_node] = router
        return self

    def run(self, initial_state: dict, max_steps: int = 20) -> dict:
        state = dict(initial_state)
        current = self.entry
        trace = []  # 실행 경로 로그 — 어떤 노드를 몇 번 거쳤는지 나중에 디버깅할 때 이 흔적이 중요하다

        # max_steps: step4의 max_steps와 똑같은 발상의 안전장치. 조건부 엣지가 잘못 짜이면
        # 같은 노드를 무한히 왔다갔다 할 수 있어서, 강제 종료 상한선이 필요하다.
        for _ in range(max_steps):
            if current is None or current == END:
                break
            trace.append(current)
            state = self.nodes[current](state)  # 노드 = "상태를 받아서 갱신된 상태를 돌려주는 순수 함수"

            # 조건부 엣지가 등록된 노드면 router(state)의 반환값으로 다음 노드를 정한다 (분기).
            # 아니면 add_edge로 미리 정해둔 고정 경로를 따른다.
            if current in self.conditional_edges:
                current = self.conditional_edges[current](state)
            else:
                current = self.edges.get(current, END)

        state["_trace"] = trace
        return state


if __name__ == "__main__":
    # 아주 단순한 예: 숫자를 받아서 짝수면 2로 나누고, 홀수면 3을 곱하는 걸 5 미만이 될 때까지 반복
    def check(state: dict) -> dict:
        return state

    def halve(state: dict) -> dict:
        state["n"] = state["n"] // 2
        return state

    def triple_plus_one(state: dict) -> dict:
        state["n"] = state["n"] * 3 + 1
        return state

    def router(state: dict) -> str:
        if state["n"] < 5:
            return END
        return "halve" if state["n"] % 2 == 0 else "triple_plus_one"

    g = Graph()
    g.add_node("check", check).add_node("halve", halve).add_node("triple_plus_one", triple_plus_one)
    g.set_entry("check")
    g.add_conditional_edge("check", router)
    g.add_edge("halve", "check")
    g.add_edge("triple_plus_one", "check")

    result = g.run({"n": 20})
    print("최종 상태:", result)
    print("거쳐간 노드:", " → ".join(result["_trace"]))
