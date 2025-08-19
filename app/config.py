# app/config.py
import os
from pathlib import Path

# 전역 기본값 (환경변수 없어도 동작)
DEFAULT_EMBED_DIM = 1536

# 데이터 디렉터리 자동 생성 (환경변수 없으면 ./data)
DATA_DIR = Path(os.getenv("FAISS_DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 인덱스 파일 경로도 자동 생성 규칙으로 결정
FAISS_INDEX_PATH = DATA_DIR / f"faiss_dim{DEFAULT_EMBED_DIM}.index"
