from fastapi import APIRouter, Depends, HTTPException, Request, Query, Body
from typing import Literal
from app.schemas.faiss_schema import (
    FaissUpsertRequest, FaissUpsertResponse, FaissReloadResponse,
    FaissStatsResponse, FaissRemoveResponse, FaissListResponse
)
import json
from app.services.faiss_service import FaissService
from app.schemas.common_response import CommonResponse

router = APIRouter()

def get_faiss(request: Request) -> FaissService:
    svc = getattr(request.app.state, "faiss", None)
    if svc is None:
        raise HTTPException(status_code=500, detail="FAISS service not initialized")
    return svc

@router.post("/upsert", response_model=CommonResponse[FaissUpsertResponse])
async def upsert(req: FaissUpsertRequest, svc: FaissService = Depends(get_faiss)):
    try:
        result = svc.upsert(req.post_id, req.embedding)
        return CommonResponse(status="success", data=FaissUpsertResponse(**result))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

@router.post("/reload", response_model=CommonResponse[FaissReloadResponse], tags=["faiss"])
async def reload_raw(
    source: Literal["jsonl", "json"] = Query("jsonl", description="jsonl or json(array)"),
    body: str = Body(
        ..., media_type="text/plain",
        description=(
            "source=jsonl → newline-delimited JSON (NDJSON)\n"
            "source=json  → JSON array\n"
            'Each item: {"postId" or "post_id", "embedding": [float,...]}'
        ),
        examples={
            "jsonl": {
                "summary": "NDJSON",
                "value": '{"postId":1,"embedding":[0.1,0.2]}\n{"postId":2,"embedding":[0.3,0.4]}'
            },
            "json": {
                "summary": "JSON array",
                "value": '[{"postId":1,"embedding":[0.1,0.2]},{"postId":2,"embedding":[0.3,0.4]}]'
            }
        },
    ),
    svc: FaissService = Depends(get_faiss),
):
    try:
        text = body

        if source == "jsonl":
            def pairs():
                for ln, line in enumerate(text.splitlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise HTTPException(status_code=400, detail=f"JSONL parse error at line {ln}: {e.msg}")

                    pid = obj.get("postId", obj.get("post_id"))
                    emb = obj.get("embedding")
                    if pid is None:
                        raise HTTPException(status_code=400, detail=f"Missing postId/post_id at line {ln}")
                    if emb is None:
                        raise HTTPException(status_code=400, detail=f"Missing embedding at line {ln}")

                    yield int(pid), emb

            result = svc.reload_from_iter(pairs())

        else:  # source == "json"
            try:
                arr = json.loads(text)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"JSON parse error: {e.msg}")
            if not isinstance(arr, list):
                raise HTTPException(status_code=400, detail="JSON payload must be an array for source=json")

            def pairs():
                for idx, obj in enumerate(arr, 1):
                    pid = obj.get("postId", obj.get("post_id"))
                    emb = obj.get("embedding")
                    if pid is None:
                        raise HTTPException(status_code=400, detail=f"Missing postId/post_id at index {idx}")
                    if emb is None:
                        raise HTTPException(status_code=400, detail=f"Missing embedding at index {idx}")
                    yield int(pid), emb

            result = svc.reload_from_iter(pairs())

        return CommonResponse(status="success", data=FaissReloadResponse(**result))

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"reload failed: {e}")

@router.get("/stats", response_model=CommonResponse[FaissStatsResponse])
async def stats(svc: FaissService = Depends(get_faiss)):
    return CommonResponse(status="success", data=FaissStatsResponse(dim=svc.dim, ntotal=svc.ntotal, index_path=svc.index_path))

@router.delete("/remove/{post_id}", response_model=CommonResponse[FaissRemoveResponse], tags=["faiss"])
async def remove(post_id: int, svc: FaissService = Depends(get_faiss)):
    try:
        removed_count = svc.remove(post_id)
        return CommonResponse(
            status="success",
            data=FaissRemoveResponse(removed=removed_count, ntotal=svc.ntotal)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"remove failed: {e}")

@router.get(
    "/ids",
    response_model=CommonResponse[FaissListResponse],
    tags=["faiss"]
)
async def list_ids(
    source: Literal["memory", "disk"] = Query("memory", description="조회 대상: memory 또는 disk"),
    offset: int = Query(0, ge=0),
    limit: int = Query(1000, gt=0),
    svc: FaissService = Depends(get_faiss),
):
    try:
        if source == "disk":
            ids = svc.get_ids_from_disk()
            ntotal = len(ids)
        else:
            ids = svc.get_ids_memory()
            ntotal = svc.ntotal

        sliced = ids[offset: offset + limit]
        return CommonResponse(
            status="success",
            data=FaissListResponse(
                dim=svc.dim,
                ntotal=ntotal,
                ids=sliced,
                source=source
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"list ids failed: {e}")