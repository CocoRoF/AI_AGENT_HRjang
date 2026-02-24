# `part1` (새 버전) vs `part1_old` (이전 버전) — 핵심 차이 요약

## 1. 🖼️ 이미지/시각화 처리 로직 전면 제거 (가장 큰 변화)

| 항목 | `part1_old` | `part1` (새 버전) |
|---|---|---|
| 이미지 다운로드 | `_download_files()`, `_list_container_file_ids()` 메서드 존재 | ❌ 완전 제거 |
| `./files/` 디렉토리 | `_create_file_directory()`로 생성 | ❌ 제거 |
| `run()` 반환값 | `(text, file_names)` 튜플 | `text`만 반환 (문자열) |
| 응답 파싱 | `parse_response()`, `normalize_image_path()`, `display_content()` 존재 | ❌ 전부 제거, `st.write()`로 직접 출력 |
| `<img>` 태그 처리 | 정규식으로 이미지 경로를 추출하여 `st.image()` 표시 | ❌ 불필요 |

**핵심 의미**: 새 버전은 **텍스트 전용 분석 에이전트**로 단순화되어, 그래프/차트 이미지 생성·다운로드·표시 로직이 모두 없어졌습니다.

---

## 2. 🔧 `tools/code_interpreter.py` 간소화

```diff
 # part1_old: 튜플 반환 + JSON 직렬화
- text_result, file_names = _code_interpreter_client.run(code)
- if file_names:
-     return json.dumps([text_result, file_names], ensure_ascii=False)
- else:
-     return json.dumps([text_result, []], ensure_ascii=False)

 # part1: 텍스트만 반환
+ text_result = _client.run(code)
+ return text_result
```

- 함수명도 `set_code_interpreter_client` → `set_client`로 짧아짐
- `json` import도 제거

---

## 3. 📝 시스템 프롬프트 대폭 축소 (57줄 → 24줄)

| `part1_old` | `part1` |
|---|---|
| matplotlib 시각화 코드 예시 포함 | 시각화 코드 없음 |
| `plt.savefig('./files/output.png')` 패턴 안내 | ❌ 제거 |
| `<img src="...">` 태그 출력 지시 | ❌ 제거 |
| seaborn 오류 → matplotlib 대체 안내 | ❌ 제거 |
| **"그래프 해석/설명 금지"** | **"그래프·차트 시각화 이미지 생성 금지"** + 표 형태로 대체 |

---

## 4. 🏠 `main.py`의 간소화

- **환영 메시지 제거**: `part1_old`는 초기화 시 `welcome_message`를 `messages`에 추가했지만, 새 버전은 빈 리스트 `[]`로 시작
- **`display_content()` → `st.write()`**: 이미지 파싱 없이 텍스트를 바로 출력
- **`main_handler.py` 파일 삭제**: `StreamlitLanggraphHandler`를 사용한 스트리밍 응답 버전이 새 버전에서는 없어짐

---

## 5. 📁 파일 구조 비교

```
part1_old/                      part1/
├── main.py                     ├── main.py
├── main_handler.py  ❌ 삭제     │
├── prompt/                     ├── prompt/
│   ├── system_prompt.txt       │   └── system_prompt.txt
│   └── system_prompt_old.txt   │
├── src/                        ├── src/
│   └── code_interpreter.py     │   └── code_interpreter.py  (92줄, 축소)
│       (148줄)                 │
└── tools/                      └── tools/
    └── code_interpreter.py         └── code_interpreter.py  (36줄, 축소)
        (45줄)
```

---

## 한 줄 요약

> **`part1`은 `part1_old`에서 이미지 생성/다운로드/표시 관련 로직을 전부 제거하여, 초보자가 이해하기 쉬운 "텍스트 전용 데이터 분석 에이전트"로 단순화한 버전입니다.**
