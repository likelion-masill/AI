# app/routers/summarize_router.py
from fastapi import APIRouter, HTTPException
from app.schemas.summarize_schema import SummarizeIn
from app.schemas.common_response import CommonResponse
from app.services.summarize_service import textrank_tfidf, refine_gpt

router = APIRouter(tags=["summarize"])

@router.post("/summarize", response_model=CommonResponse)
def summarize(body: SummarizeIn):
    if len(body.text) < 10:
        raise HTTPException(status_code=400, detail="입력 텍스트가 너무 짧음")

    # Step 1: 추출 요약 (TextRank/TF-IDF)
    extractive = textrank_tfidf(body.text, body.top_k, body.min_len)

    # Step 2: 생성 요약 (GPT 호출)
    try:
        data = refine_gpt(
            draft=extractive,
            temperature=body.temperature,
            max_tokens=body.max_output_tokens,
        )
        status = "success"
    except Exception:
        data = None
        status = "fail"

    return CommonResponse(status= status, data= data)
