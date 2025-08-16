# app/routers/summarize.py
from fastapi import APIRouter, HTTPException
from app.schemas.summarize import SummarizeIn, SummarizeOut
from app.services.summarizer import textrank_tfidf, refine_gpt

router = APIRouter(tags=["summarize"])

@router.post("/summarize", response_model=SummarizeOut)
def summarize(body: SummarizeIn):
    if len(body.text) < 10:
        raise HTTPException(status_code=400, detail="입력 텍스트가 너무 짧음")

    # Step 1: 추출 요약
    extractive = textrank_tfidf(body.text, body.top_k, body.min_len)

    # Step 2: 생성 요약 (GPT 호출, 실패 시 폴백)
    try:
        abstractive = refine_gpt(
            draft=extractive,
            temperature=body.temperature,
            max_tokens=body.max_output_tokens,
        )
        status = "DONE"
    except Exception:
        abstractive = ""
        status = "PARTIAL"

    return SummarizeOut(extractive=extractive, abstractive=abstractive, status=status)
