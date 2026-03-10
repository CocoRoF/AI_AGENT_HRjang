다음과 같은 기준으로 검토를 진행했습니다.
1. 기존 코드가 작동하면 최대한 로직을 훼손하지 않음.
2. 현재 설명에 "치명적 오류" 가 없으면 해당 설명을 변경하지 않음.
3. 추가적 설명이 필요할 수도 있다면 검토 의견만 제시.

-- 표지 --

수정 요청:
재미있어 보이는 것은 일단 만들고 보는 AI 엔지니어 이자 오픈소스 개발자. 언어 모델과 컴퓨터 비전 모델을 중심으로 다양한 경험과 기술을 쌓아가고 있다. 특히 LangChain, LangGraph, ComfyUI, OpenClaw 등 에이전트 기반 기술과 생태계에 깊은 관심을 두고 있으며, 현재 (주)플래티어 AI-Lab에서 AI 에이전트 플랫폼 개발을 리드하고 있다.
경희대학교 대학원에서 빅데이터와 인공지능을 전공한 뒤 이를 실무에서 현실화 시키는 일을 하고 있다. 지식을 학문에만 머물게 두지 않고 일단 만들고 보는 AI 엔지니어이자 오픈소스 개발자. 특히 LangChain, LangGraph, ComfyUI, OpenClaw 등 에이전트 기반 기술과 생태계에 깊은 관심을 두고 있으며, 현재 (주)플래티어 AI-Lab에서 AI 에이전트 플랫폼 개발을 리드하고 있다.

-- 공통 --
_init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.0.1)/charset_normalizer (3.4.5) doesn't match a supported version!
  warnings.warn(

현재 실행시 위와 같은 경고 메세지 출력됨.

requests==2.32.5
readability-lxml==0.8.4.1

상기 2가지 라이브러리의 버전에 의한 이슈이나 ... 실제로 동작 로직에는 큰 문제가 없음.

이유는 requests는 다른 라이브러리를 통해 정상 작동 (충돌하는 라이브러리 사용 안 함)
readability-lxml는 충돌하는 해당 라이브러리를 이용해 정상 작동 (서로 관여하지 않는 구조)

--> 따라서 해당 사항은 따로 조치하지 않았습니다. 버전 변경에 따른 이슈 발생 가능성이 더 크기 때문입니다.


----- 1장 -----
(공유해 주신 코드 부분 chap1의 main.py 관련 에러만 수정)

main.py 정상 작동 확인 (OpenAI 사용)
my_first_app.py 작동 확인

- 14p, 35p
리마인드: 랭체인 버전 일치하도록 수정할 것.

- 18p, 19p, 20p
검토:
리눅스/맥의 경우 환경변수 설정을
export OPENAI_API_KEY="sk-f302ur02h932pjhf0oqahfefujikofnaljf..."  이와 같은 방식으로 진행

windows의 경우에는 cli (cmd / powershell 모두)에서
set OPENAI_API_KEY=sk-f302ur02h932pjhf0oqahfefujikofnaljf... 이와 같은 방식으로 진행해야만 함. (큰 따옴표 쓰지 않음)


----- 2장 -----
main.py 정상 작동 확인 (OpenAI 사용)

전체적으로 수정할 부분 없음.
streamlit과 관련된 부분은 기존 내용 그대로 사용(버전 이슈가 없으면 그대로 동작)
LangChain의 기본 Method는 그대로 유지. 작동만 확인.


----- 3장 -----
main.py 정상 작동 확인 (OpenAI 사용)

- 85p
수정 제안:
get_message_counts 함수 부분을 아래와 같이 수정하였습니다.

def get_message_counts(text):
    if "gemini" in st.session_state.model_name:
        return st.session_state.llm.get_num_tokens(text)
    else:
        try:
            encoding = tiktoken.encoding_for_model(st.session_state.model_name)
        except KeyError:
            # tiktoken이 인식하지 못하는 모델(ex: Claude 모델)은 gpt-4o 인코딩 사용
            encoding = tiktoken.encoding_for_model("gpt-4o")
        return len(encoding.encode(text))

기존 코드는 gpt라는 keyword를 통해 gpt 모델을 사용하려고 시도하는데
gpt-5.1 같은 모델의 경우 tiktoken에서 지원하지 않기 때문에 에러가 발생.

따라서 Try-Except 구조로 입력된 값을 시도하고, 에러 발생시 gpt-4o를 사용하도록 변경.

이외 다른 부분 문제 없음.


----- 4장 -----
- 95p
리마인드: 랭체인 버전 일치하도록 수정할 것.

streamlit과 관련된 부분은 수정하지 않았습니다.


----- 5장 -----
part1: main.py 정상 작동 확인 (OpenAI 사용)
part2: main.py 정상 작동 확인 (OpenAI 사용)

part2의 map_reduce.py 파일의 코드만 일부 수정.

-- 기존 --
        except:
            st.error(f"Error occurred: {e}")
            st.write(traceback.format_exc())
            return None

-- 수정 --
        except Exception as e:
            st.error(f"Error occurred: {e}")
            st.write(traceback.format_exc())
            return None

기존 에러 발생시 except 구분에 e로 받아주는 것이 없어
다음 코드인 st.error 라인에서 에러가 발생. 해당 문제만 강건하게 수정.


----- 6장 -----
dalle_gpt-image-1.py 정상 작동 확인 (OpenAI 사용)
gpt5v.py 정상 작동 확인 (OpenAI 사용)
ttl.py 정상 동작 확인

- 162p
검토:
st.image 함수의 경우 use_container_width 파라미터를 더 이상 사용하지 않는다고 명시하고 있음.
(최신 streamlit을 사용시 해당 파라미터는 이미 제거됨: 25.12.31 이후)

-- 기존 --
    # 생성된 이미지 표시
    if generated_image_base64:
        st.markdown("### Question")
        st.write(user_input)
        st.image(uploaded_file, use_container_width=True)
        st.markdown("### Generated Image")
        # base64를 디코딩하여 이미지로 표시
        image_bytes = base64.b64decode(generated_image_base64)
        st.image(image_bytes, caption=image_prompt, use_container_width=True)

-- 수정 --
    if generated_image_base64:
        st.markdown("### Question")
        st.write(user_input)
        st.image(uploaded_file, width="stretch")
        st.markdown("### Generated Image")
        # base64를 디코딩하여 이미지로 표시
        image_bytes = base64.b64decode(generated_image_base64)
        st.image(image_bytes, caption=image_prompt, width="stretch")

따라서 해당 코드에서 아래와 같이 use_container_width 를 width로 수정하고
기존 True 값에 해당하는 "stretch" 인자를 주어 처리하였음.


----- 7장 -----
main.py 정상 작동 확인 (OpenAI 사용, 하위 페이지 모두 정상 동작)


----- 8장 -----
-207p
리마인드: 랭스미스 버전 일치하도록 수정할 것.
