from openai import OpenAI

client = OpenAI()

with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="nova",
    instructions="활기차고 믿음직한 출판사 홍보 톤으로",
    input="여기는 여러분의 지식을 업그레이드하는 곳, 영진닷컴입니다! 오늘도 우리와 함께 한 단계 성장해볼까요?",
) as response:
    response.stream_to_file("speech.mp3")


# 코드 변경의 핵심 이유
#
# stream_to_file() deprecated - 실제로 스트리밍 안 되는 버그 있음
# with_streaming_response 사용 - 올바른 스트리밍 방식
# gpt-4o-mini-tts 모델 - 2025년 3월 출시, 더 자연스럽고 스타일 제어 가능
# instructions 파라미터 추가 - 음성 톤/스타일 자연어로 지정 가능 (새 기능)
# 음성 옵션 확장 - 기존 6개 → 11개 이상으로 증가
