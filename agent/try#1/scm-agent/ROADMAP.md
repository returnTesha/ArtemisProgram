# SCM Agent 학습 로드맵 (Step 1 ~ 30)

*(English version: [ROADMAP.en.md](./ROADMAP.en.md))*

목표: 프레임워크(LangChain 등) 없이 raw OpenAI SDK + 로컬 Ollama(qwen2.5:7b, bge-m3)로
"AI agent가 내부적으로 어떻게 동작하는지"를 바닥부터 쌓은 뒤, 마지막에 실제 프레임워크(LangGraph)로
같은 걸 다시 구현해보며 비교한다. 도메인은 반도체 회사 SCM(발주/재고/납기) 업무 도우미로 고정.

모든 예제는 `client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")` 로
로컬 Ollama를 OpenAI 호환 API처럼 사용한다.

## 완료 (Step 1~5)

- **Step 1 — LLM 호출 기초**: 단일 호출, 멀티턴 대화, stateless 특성(대화 기록을 매번 다 보내야 하는 이유), temperature
- **Step 2 — 구조화된 출력**: 프롬프트로만 JSON 요청 → 실패 사례 → JSON Schema 강제 → Pydantic 검증 → 필드 순서(reason-first)가 정확도에 미치는 영향
- **Step 3 — Tool calling 기초**: 함수 명세 작성, LLM의 tool_calls 파싱, 실행 결과를 다시 대화에 넣어 최종 답변 생성
- **Step 4 — Agent loop**: 여러 도구를 여러 스텝에 걸쳐 순차 호출, max_steps 안전장치, 처리 불가 요청 안내
- **Step 5 — Embedding**: 텍스트 → 벡터, 코사인 유사도로 의미 기반 비교 (RAG의 기초)

## 신규 (Step 6~20)

- **Step 6 — RAG 기초 (직접 구현)**: 문서를 chunk로 쪼개고, 전부 메모리에 올려 임베딩 후 코사인 유사도로 검색 → 프롬프트에 컨텍스트로 주입. 프레임워크나 벡터DB 없이 RAG의 본질(retrieve → augment → generate)을 이해
- **Step 7 — 벡터DB (ChromaDB)**: 실제 벡터DB에 저장/검색. 메모리 방식과 달리 영속성, 대량 문서, 메타데이터 필터링이 왜 필요한지
- **Step 8 — 검색 품질 향상**: 키워드 검색(BM25) vs 임베딩 검색의 차이, 두 방식을 합치는 하이브리드 검색, LLM을 이용한 재랭킹(rerank)
- **Step 9 — Agent + RAG 결합**: RAG 검색 자체를 하나의 tool로 등록해서, "지금 검색이 필요한지"를 LLM이 스스로 판단하게 만듦 (step4 도구들과 통합)
- **Step 10 — 메모리**: 대화가 길어질 때 요약으로 압축하는 단기 메모리, 과거 대화를 벡터로 저장해뒀다가 유사한 대화를 다시 꺼내오는 장기 메모리
- **Step 11 — 추론 패턴 (ReAct / Plan-Execute)**: Thought → Action → Observation을 명시적으로 반복하는 ReAct, 먼저 계획을 다 세우고 나서 실행하는 Plan-and-Execute의 차이
- **Step 12 — 멀티 에이전트 (Supervisor)**: 재고 전문가/납기 전문가처럼 역할을 쪼갠 에이전트들과, 요청을 보고 누구에게 맡길지 정하는 supervisor
- **Step 13 — 그래프 오케스트레이션 (직접 구현)**: 노드/엣지/조건부 분기로 이루어진 미니 그래프 실행기를 직접 만들어 SCM 워크플로우를 표현. "그래프 AI"가 실제로 무엇을 하는지 프레임워크 없이 이해
- **Step 14 — LangGraph**: 13에서 직접 만든 그래프를 실제 LangGraph 라이브러리로 재구현하고, prebuilt ReAct agent와 비교
- **Step 15 — 자율 루프 & Reflection**: 답변을 스스로 비평하고 개선하는 self-reflection, 목표만 주어졌을 때 스스로 다음 행동과 종료 시점을 정하는 자율 루프("루프 AI")
- **Step 16 — 스트리밍**: `stream=True`로 토큰 단위 실시간 출력, 스트리밍 중 tool call을 처리하는 방법
- **Step 17 — 가드레일 & 견고성**: 입력 검증(공격/오용 차단), 출력 필터링, 재시도/타임아웃/예외 처리로 프로덕션에서 안 죽는 에이전트 만들기
- **Step 18 — 평가 (Evaluation)**: LLM-as-judge로 답변 품질을 자동 채점하고, 여러 케이스를 한 번에 돌리는 평가 파이프라인
- **Step 19 — Observability**: 각 단계 실행을 로깅/추적하는 데코레이터, 토큰 사용량과 지연시간 같은 메트릭 수집
- **Step 20 — 최종 통합 프로젝트**: RAG + Tool + Memory + Graph 오케스트레이션 + 가드레일 + 추적을 전부 합친 SCM 최종 에이전트

## 백엔드 심화 (Step 21~30)

1~20이 "에이전트 내부 동작 원리"에 집중했다면, 21~30은 그걸 실제 백엔드 서비스로 만들 때 마주치는 문제들이다.

- **Step 21 — MCP 기초**: tool 정의와 실행을 별도 프로세스(서버)로 분리하는 Model Context Protocol. 실제 MCP 서버를 만들고, list_tools/call_tool을 직접 호출해본다
- **Step 22 — MCP + Agent 통합**: agent가 tool 목록과 실행을 전부 MCP 서버에게 위임 — 에이전트 코드가 tool의 구체적인 구현을 전혀 몰라도 동작하게 만든다
- **Step 23 — Human-in-the-loop**: 되돌리기 어려운 도구(실제 발주)는 사람 승인을 거치게 한다. input()으로 그 자리에서 막는 방식과, LangGraph의 interrupt()로 실행 자체를 중단했다가 나중에(다른 프로세스에서도) 재개하는 방식 둘 다 구현
- **Step 24 — 프롬프트 캐싱**: 반복되는 긴 컨텍스트를 앞에 고정 배치해서 재계산을 줄이는 원리. 로컬 모델의 체감 캐싱과, 클라우드 API의 명시적 cache_control 차이
- **Step 25 — 비용 기반 모델 라우팅**: 쉬운 질문은 작은/싼 모델, 어려운 질문은 큰/비싼 모델로. 실제로 작은 모델에게 난이도 판단을 시켜보면 왜 신뢰할 수 없는지, 그래서 왜 규칙 기반 라우팅이 필요한 경우가 많은지
- **Step 26 — 컨텍스트 압축**: 대화가 길어질수록 커지는 건 대화 턴이 아니라 tool 호출 결과다. 오래된 tool 결과를 한 줄 요약으로 압축해서 컨텍스트 예산을 관리
- **Step 27 — 프레임워크 비교**: 직접 만든 supervisor(step12)와 AutoGen의 SelectorGroupChat이 같은 작업을 어떻게 다르게 처리하는지 — 코드로 라우팅을 확정하는 것과 LLM이 매번 라우팅을 판단하는 것의 실제 차이
- **Step 28 — 보안 레드티밍**: step17 가드레일을 공격자 입장에서 깨보기. 직접 명령/역할극 탈옥/우회 표기 등 실제 우회 사례를 찾고 고치는 반복 과정
- **Step 29 — FastAPI 서빙 + 배포**: 스크립트로 실행하던 에이전트를 HTTP API로 감싸고, Dockerfile로 컨테이너화. 컨테이너 안에서 호스트의 Ollama에 접근하는 실전 이슈(host.docker.internal)까지 포함
- **Step 30 — 최종 통합 프로젝트 v2**: MCP 기반 tool + human-in-the-loop 승인 + 그래프 오케스트레이션 + 가드레일 + 추적을 전부 합친, 실제 서비스에 가장 가까운 형태의 최종 에이전트

## 진행 방식

각 스텝은 `stepN/` 폴더 아래에 개념별 작은 스크립트 여러 개로 구성된다 (기존 step1~5와 동일한 패턴).
직접 실행해보고, 출력을 보고, 코드를 수정해가며 왜 그렇게 동작하는지 체감하는 것이 목적이다.
