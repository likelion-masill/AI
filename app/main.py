# app/main.py
from fastapi import FastAPI
from app.routers.summarize_router import router as summarize_router
from app.routers.health_router import router as health_router
from app.routers.faiss_router import router as faiss_router
from app.services.faiss_service import FaissService
from app.config import DEFAULT_EMBED_DIM, FAISS_INDEX_PATH

def create_app() -> FastAPI:
    app = FastAPI(title="masill AI API", version="0.1.0")

    app.include_router(health_router,     prefix="/api",       tags=["health"])
    app.include_router(summarize_router,  prefix="/api",       tags=["summarize"])
    app.include_router(faiss_router,      prefix="/api/faiss", tags=["faiss"])

    @app.on_event("startup")
    async def _startup():
        dim = DEFAULT_EMBED_DIM
        index_path = str(FAISS_INDEX_PATH)  # pathlib → str
        app.state.faiss = FaissService(dim=dim, index_path=index_path,
                                       use_cosine=True, autosave=True)
        loaded = app.state.faiss.load_index_if_exists()
        print(f"[FAISS] startup: dim={dim}, loaded={loaded}, path={index_path}")

    return app

app = create_app()