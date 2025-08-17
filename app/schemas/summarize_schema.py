from pydantic import BaseModel

class SummarizeIn(BaseModel):
    text: str                     # 요약할 원본 텍스트
    top_k: int = 5                # 추출 요약 시 뽑을 문장 개수
    min_len: int = 10             # 요약 문장의 최소 길이
    temperature: float = 0.3      # 생성 요약(LLM 호출) 시 다양성 조절
    max_output_tokens: int = 300  # 생성 요약의 최대 토큰 수
