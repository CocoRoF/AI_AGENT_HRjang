# 수정사항 기록

### 수정 완료

1. requirements.txt : youngjin-langchain-tools==0.3.4 로 버전 변경 (호환성 이슈 해결)
2. chapter_001\main.py (23행): `MessagesPlaceholder(variable_name="history")("user", "{user_input}")` → 쉼표 누락으로 TypeError 발생. `MessagesPlaceholder(variable_name="history"), ("user", "{user_input}")` 로 수정.
3. chapter_003\main.py (95~103행): `tiktoken.encoding_for_model("gpt-5.1")` 호출 시 KeyError 크래시. tiktoken 0.12.0이 gpt-5.1 모델명을 인식하지 못함. `try/except KeyError` fallback으로 `tiktoken.get_encoding("o200k_base")` 사용하도록 수정.
4. chapter_005\part2\map_reduce.py (130~131행): `except:` bare except 블록에서 미정의 변수 `e`를 참조하여 NameError 발생. `except Exception as e:` 로 수정.
5. chapter_009\tools\search_ddgs.py : backend 파라미터를 auto로 변경, `DDGSException` import 및 try/except 에러 핸들링 추가. DDGS 자체에서 검색 결과에 대한 에러가 발생하는 경우가 존재. 현재 원인으로는 duckduckgo 엔진의 경우 짧은 시간 반복 요청을 차단하고 있는 것으로 파악. LLM이 매우 짧은 시간에 연속된 검색 호출 (LLM이 도구를 순차적 사용이 아니라, 병렬적 검색을 수행)하여 생기는 것으로 판단.
6. chapter_009\main.py : SummarizationMiddleware 파일의 수정이 반영되지 않아서, 수정을 반영해 두었습니다. (1.2 버전 업그레이드 이유)
7. chapter_010\vectorstore\qa_vectorstore: `index.pkl` 파일 누락으로 `FAISS.load_local()` 시 FileNotFoundError 발생. `build_qa_vectorstore.py` 재실행으로 복구. (cache 디렉터리도 동일하게 `index.pkl` 누락이나, 코드상 첫 save() 시 자동 생성되므로 문제 없음)

### 조건부 에러 (기록만, 미수정)

1. chapter_001\main.py, chapter_002\st.container.py: `ChatOpenAI(temperature=0)` — 모델 미지정. 기본값 gpt-3.5-turbo 사용됨. 의도적이라면 무방.
2. chapter_003\main.py: Gemini 모델 선택 시 `llm.get_num_tokens(text)` 호출에 `GOOGLE_API_KEY` 필요. 미설정 시 에러 발생.
3. chapter_006\dalle_gpt-image-1.py (63행): LLM 프롬프트 생성에 `model="gpt-4o"` 사용. 다른 챕터는 gpt-5.x 계열. 의도적 선택인지 확인 필요.
4. chapter_009\main_handler.py: `youngjin_langchain_tools.StreamlitLanggraphHandler` 외부 패키지 의존. API 변경 시 영향 있음.
5. chapter_010 전체: 상대 경로(`./prompt/`, `./data/`, `./vectorstore/`) 사용. 반드시 `cd chapter_010` 후 `streamlit run main.py` 실행 필요.
6. chapter_010\main_feedback.py: LangSmith 환경변수(`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`) 미설정 시 feedback 전송 실패.
7. chapter_011\part1\main.py: OpenAI Containers API 기능 활성화 필요.
8. chapter_011\part2\main.py: `.streamlit/secrets.toml` 파일 누락. GCP 서비스 계정 JSON 설정 필요. `project_id` 하드코딩되어 있으므로 독자 본인의 프로젝트 ID로 변경 필요.
9. chapter_011\part2\tools\bigquery.py: `table_name`이 f-string으로 SQL에 직접 삽입됨. 실질적 위험은 낮으나 파라미터화 권장.
10. 프로젝트 전체: `.env` 파일 부재. `.env.template`만 존재하므로 API 키를 환경변수로 별도 설정해야 함.

### 권장사항 (기록만, 미수정)

1. chapter_001\main.py, chapter_002\st.container.py: message_history 형식이 튜플(`("user", text)`)로 되어 있으나, chapter_002\main.py부터 dict 형식 사용. 일관성 차원 검토 필요.
2. chapter_001\token_test copy.py: 파일명에 공백 포함. 일부 도구/CI에서 문제 될 수 있음.
3. chapter_005\part1\main.py: `requests.get(url)` timeout 미지정. 무응답 서버 시 무한 대기 가능.
4. chapter_005\part1\main.py (79행), chapter_009\tools\fetch_page.py (68행): bare except 사용. 디버깅 불편하나 기능 문제 없음.
5. chapter_009\tools\fetch_page.py: `from_tiktoken_encoder(model_name="gpt-3.5-turbo")` — 구모델이나 토크나이저 용도이므로 기능 문제 없음.
6. chapter_011\part1\main.py (1행): `import re` 미사용. 불필요한 import.
7. 프로젝트 전체: `chardet 7.0.1`이 `requests 2.32.5`의 허용 범위(`< 6.0.0`) 초과. 기능 영향 없으나 RequestsDependencyWarning 경고 출력됨.
