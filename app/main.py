# app/main.py
from fastapi import FastAPI
from app.routers.summarize import router as summarize_router
from app.routers.health import router as health_router

app = FastAPI(title="Summarizer API", version="0.1.0")

# 라우터 등록
app.include_router(summarize_router, prefix="/api")
app.include_router(health_router, prefix="/api")