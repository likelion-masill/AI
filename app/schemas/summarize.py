from pydantic import BaseModel

class SummarizeIn(BaseModel):
    text: str
    top_k: int = 5
    min_len: int = 10
    temperature: float = 0.3
    max_output_tokens: int = 300

class SummarizeOut(BaseModel):
    extractive: str
    abstractive: str
    status: str