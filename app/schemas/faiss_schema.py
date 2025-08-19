import os
from typing import Annotated, List, Literal
from pydantic import BaseModel, Field
from app.config import DEFAULT_EMBED_DIM as EMBED_DIM

class FaissUpsertRequest(BaseModel):
    post_id: int
    embedding: Annotated[List[float], Field(min_length=1)]

class FaissUpsertResponse(BaseModel):
    upserted: int
    ntotal: int

class FaissReloadResponse(BaseModel):
    loaded: int
    ntotal: int

class FaissStatsResponse(BaseModel):
    dim: int
    ntotal: int
    index_path: str

class FaissRemoveResponse(BaseModel):
    removed: int
    ntotal: int

class FaissListResponse(BaseModel):
    dim: int
    ntotal: int          # 전체 개수 (source에 따라 메모리 or 디스크 기준)
    ids: List[int]       # 페이징된 id 목록
    source: Literal["memory", "disk"]