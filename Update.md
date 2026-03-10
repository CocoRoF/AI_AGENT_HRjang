# 수정사항 기록

### 수정 완료

1. (중요) requirements.txt : youngjin-langchain-tools==0.3.4 로 버전 변경 (호환성 이슈 해결)
2. (Minor, 1장 main.py 본문에서 사용하지 않음) chapter_001\main.py (23행): `MessagesPlaceholder(variable_name="history")("user", "{user_input}")` → 쉼표 누락으로 TypeError 발생. `MessagesPlaceholder(variable_name="history"), ("user", "{user_input}")` 로 수정.

3. (중요) chapter_003\main.py (95~103행): `tiktoken.encoding_for_model("gpt-5.1")` 호출 시 KeyError 크래시. Try - Except 구조로 변경해서 조금 더 강건하게 수정. 기존에는 if - else 구조로 문제가 발생하였음.

4. chapter_005\part2\map_reduce.py (130~131행): `except:` bare except 블록에서 미정의 변수 `e`를 참조하여 NameError 발생. `except Exception as e:` 로 수정.
5. chapter_009\tools\search_ddgs.py : backend 파라미터를 auto로 변경, `DDGSException` import 및 try/except 에러 핸들링 추가. DDGS 자체에서 검색 결과에 대한 에러가 발생하는 경우가 존재. 현재 원인으로는 duckduckgo 엔진의 경우 짧은 시간 반복 요청을 차단하고 있는 것으로 파악. LLM이 매우 짧은 시간에 연속된 검색 호출 (LLM이 도구를 순차적 사용이 아니라, 병렬적 검색을 수행)하여 생기는 것으로 판단.
6. chapter_009\main.py : SummarizationMiddleware 파일의 수정이 반영되지 않아서, 수정을 반영해 두었습니다. (1.2 버전 업그레이드 이유)
7. chapter_010\vectorstore\qa_vectorstore: `index.pkl` 파일 누락으로 `FAISS.load_local()` 시 FileNotFoundError 발생. `build_qa_vectorstore.py` 재실행으로 복구. (cache 디렉터리도 동일하게 `index.pkl` 누락이나, 코드상 첫 save() 시 자동 생성되므로 문제 없음)
