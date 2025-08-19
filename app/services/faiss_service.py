import os
import io
import json
import faiss
import numpy as np
import threading
import tempfile
from typing import Iterable, Tuple, List, Dict, Optional


class FaissService:
    """
    Cosine 유사도를 Inner Product로 계산하기 위해 벡터를 L2 정규화하고
    IndexFlatIP + IDMap2를 사용한다. ID = postId(64-bit).
    """
    # Parameters:
    # - dim: 임베딩 벡터 차원 (예: 1536)
    # - index_path: 인덱스를 디스크에 저장할 파일 경 (예: "./data/faiss_dim1536.index")
    # - use_cosine: True면 코사인 유사도를 쓰도록 설정(내적 기반 + 정규화).
    # - autosave: True면 변경 시마다(upsert/삭제 후) 자동으로 인덱스 저장
    def __init__(self, dim: int, index_path: str, use_cosine: bool = True, autosave: bool = True):
        self.dim = dim
        self.index_path = index_path
        self.use_cosine = use_cosine
        self.autosave = autosave
        # 쓰레드 락
        self._lock = threading.RLock()
        # 메모리에 새 FAISS 인덱스 객체를 만들고 보관
        # 파일에 저장된 게 있으면 나중에 load_index_if_exists()에서 교체 로드함.
        self._index = self._new_index()
        # 인덱스 파일을 저장할 디렉터리가 없으면 생성
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # ---------- index helpers ----------
    # 빈 FAISS 인덱스(저장소) 생성
    def _new_index(self):
        base = faiss.IndexFlatIP(self.dim)  # cosine == IP on normalized vectors
        return faiss.IndexIDMap2(base)

    # 임베딩 벡터 -> 길이 1인 단위벡터로 정규화
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if not self.use_cosine:
            return X
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return X / norms

    # 디스크에 존재하는 인덱스 파일을 메모리에 로드
    def load_index_if_exists(self) -> int:
        with self._lock:
            if os.path.exists(self.index_path):
                self._index = faiss.read_index(self.index_path)
            return self.ntotal

    # 현재 인덱스를 디스크에 저장
    # .tmp 파일로 먼저 저장 → 저장 성공하면 최종 파일로 교체
    def save_index(self):
        with self._lock:
            tmp = self.index_path + ".tmp"
            faiss.write_index(self._index, tmp)
            os.replace(tmp, self.index_path)

    # ---------- public props ----------
    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    # ---------- mutating ops ----------
    # 이미 같은 postId 있으면 교체, 없으면 새로 삽입.
    def upsert(self, post_id: int, embedding: List[float]) -> Dict:
        # 임베딩 차원 검증
        if len(embedding) != self.dim:
            raise ValueError(f"dim mismatch: got {len(embedding)}, expected {self.dim}")

        # 입력 데이터 전처리
        x = np.asarray(embedding, dtype="float32").reshape(1, -1)
        x = self._normalize(x)
        ids = np.asarray([post_id], dtype="int64")

        with self._lock:
            # postId가 이미 존재한다면 제거 후 새로 추가
            self._index.remove_ids(ids)
            self._index.add_with_ids(x, ids)
            if self.autosave:
                self.save_index()
            return {"upserted": 1, "ntotal": self.ntotal}

    def remove(self, post_id: int) -> int:
        with self._lock:
            n = int(self._index.remove_ids(np.asarray([post_id], dtype="int64")))
            if self.autosave:
                self.save_index()
            return n

    # ---------- bulk reload ----------
    def reload_from_iter(self, pairs: Iterable[Tuple[int, List[float]]]) -> Dict:
        """
        pairs: iterable of (postId, embedding[list[float]])
        스트리밍으로 읽으면서 chunk 단위로 추가
        """
        # 새로운 인덱스 만든 후 기존 인덱스는 교체
        new_index = self._new_index()

        # numpy로 형변환, 정규화 -> 인덱스에 추가
        def flush(buf_ids: List[int], buf_vecs: List[List[float]]):
            if not buf_ids:
                return 0
            X = np.asarray(buf_vecs, dtype="float32")
            X = self._normalize(X)
            I = np.asarray(buf_ids, dtype="int64")
            new_index.add_with_ids(X, I)
            return len(buf_ids)

        # pairs 순회하면서 청크 단위로 추가
        chunk_ids, chunk_vecs = [], []
        count = 0
        for pid, emb in pairs:
            if len(emb) != self.dim:
                raise ValueError(f"postId {pid}: dim mismatch {len(emb)} != {self.dim}")
            chunk_ids.append(int(pid))
            chunk_vecs.append(emb)
            # 2048 : 청크 단위
            if len(chunk_ids) >= 2048:
                count += flush(chunk_ids, chunk_vecs)
                chunk_ids, chunk_vecs = [], []

        count += flush(chunk_ids, chunk_vecs)

        # 메모리상의 인덱스 교체 후 디스크에 저장
        with self._lock:
            self._index = new_index
            if self.autosave:
                self.save_index()

        return {"loaded": count, "ntotal": count}
