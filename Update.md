2026.03.07 수정사항 기록

1. requirements.txt 파일: youngjin-langchain-tools==0.3.4 로 버전 변경 (호환성 이슈 해결)
2. chapter_009\tools\search_ddgs.py 파일: backend 파라미터 제거. DDGS 자체에서 검색 결과에 대한 에러가 발생하는 경우가 존재. Backend 값을 설정하지 않으면 Auto 모드로 작동하며, 상황에 따라 작동합니다.
3. chapter_009\main.py 파일: SummarizationMiddleware 파일의 수정이 반영되지 않아서, 수정을 반영해 두었습니다.
