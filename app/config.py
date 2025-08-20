# app/config.py
import os, platform
from pathlib import Path

# 전역 기본값 (환경변수 없어도 동작)
DEFAULT_EMBED_DIM = 1536

if platform.system() == "Linux":
    DATA_DIR = Path("/var/lib/masill/faiss")
else:
    DATA_DIR = Path("./data")

DATA_DIR.mkdir(parents=True, exist_ok=True)

# 인덱스 파일 경로도 자동 생성 규칙으로 결정
FAISS_INDEX_PATH = DATA_DIR / f"faiss_dim{DEFAULT_EMBED_DIM}.index"
