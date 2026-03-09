2026.03.07 수정사항 기록

1. requirements.txt 파일: youngjin-langchain-tools==0.3.4 로 버전 변경 (호환성 이슈 해결)
2. chapter_009\tools\search_ddgs.py 파일: backend 파라미터를 auto로 변경. DDGS 자체에서 검색 결과에 대한 에러가 발생하는 경우가 존재. 현재 원인으로는 duckduckgo 엔진의 경우 짧은 시간 반복 요청을 차단하고 있는 것으로 파악. LLM이 매우 짧은 시간에 연속된 검색 호출 (LLM이 도구를 순차적 사용이 아니라, 병렬적 검색을 수행)하여 생기는 것으로 판단.
3. chapter_009\main.py 파일: SummarizationMiddleware 파일의 수정이 반영되지 않아서, 수정을 반영해 두었습니다.
