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

    # 현재 메모리 상 인덱스에서 모든 ID 가져오기
    def get_ids_memory(self) -> List[int]:
        with self._lock:
            if self.ntotal == 0:
                return []
            try:
                # IndexIDMap2의 id_map(std::vector) -> numpy로
                ids = faiss.vector_to_array(self._index.id_map)
            except AttributeError:
                # 혹시 구현 차이로 속성이 다를 경우 대비
                return []
            return ids.astype("int64").tolist()

    # 디스크에 저장된 인덱스 파일에서 모든 ID 가져오기
    def get_ids_from_disk(self) -> List[int]:
        with self._lock:
            if not os.path.exists(self.index_path):
                return []
            idx = faiss.read_index(self.index_path)
        try:
            ids = faiss.vector_to_array(idx.id_map)
            return ids.astype("int64").tolist()
        except AttributeError:
            return []

    def search_subset(
            self,
            query_embedding: List[float],
            candidate_ids: List[int],
            top_k: int = 10,
            normalize_query: bool = True,
            # 추가된 하이퍼파라미터
            tau_abs: float = 0.30,  # 최소 점수 임계값(코사인)
            delta_margin: float = 0.04,  # 상대 마진(상위 평균 대비 허용 하락폭)
    ) -> Dict:
        """
        candidate_ids 서브셋 내부에서만 쿼리 임베딩과의 유사도(내적/코사인)로 상위 top_k 랭킹 반환.
        필터 순서: 절대 임계값(τ_abs) → 상대 마진(Δ) → top-k.
        반환:
          {
            "total": <int>,   # present_ids 개수 (필터 전, top_k 자르기 전)
            "results": [{"post_id": int, "score": float}, ...]
          }
        """
        if not candidate_ids:
            return {"total": 0, "results": []}
        if len(query_embedding) != self.dim:
            raise ValueError(f"query dim mismatch: got {len(query_embedding)}, expected {self.dim}")
        if top_k <= 0:
            return {"total": 0, "results": []}

        q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        if self.use_cosine and normalize_query:
            q = self._normalize(q)
        q = q.reshape(-1)  # (d,)

        with self._lock:
            if self.ntotal == 0:
                return {"total": 0, "results": []}
            try:
                id_arr = faiss.vector_to_array(self._index.id_map)  # numpy array of ids
            except AttributeError:
                raise RuntimeError("IndexIDMap2.id_map not available on this build")

            id_set = set(int(x) for x in id_arr)

            vecs = []
            present_ids = []
            print(f"[DEBUG] 입력 candidate_ids 개수: {len(candidate_ids)}")
            for id in candidate_ids:
                print(f"[DEBUG] id : {id}")

            for pid in candidate_ids:
                pid = int(pid)
                if pid not in id_set:
                    continue
                v = self._index.reconstruct(pid)  # (d,)
                vecs.append(v)
                present_ids.append(pid)
            print(f"[DEBUG] 실제 인덱스에 존재하는 id 개수: {len(present_ids)}")

        if not present_ids:
            return {"total": 0, "results": []}

        # 유사도 계산
        M = np.stack(vecs, axis=0).astype("float32")  # (N, d)
        sims = M @ q  # (N,)

        total_before_filter = len(present_ids)

        # --- (1) 절대 임계값 필터 ---
        keep_abs = sims >= float(tau_abs)
        if not np.any(keep_abs):
            # 모두 컷나면 상위 1개는 관용적으로 통과 (추천 공백 방지)
            top_idx_all = int(np.argmax(sims))
            kept_ids = np.array([present_ids[top_idx_all]])
            kept_sims = np.array([float(sims[top_idx_all])], dtype="float32")
        else:
            kept_ids = np.array(present_ids, dtype=np.int64)[keep_abs]
            kept_sims = sims[keep_abs]

        # --- (2) 상대 마진 필터 (상위 그룹 평균 대비 Δ 허용) ---
        # 상위 5개(또는 그 미만)의 평균을 상위 평균으로 사용
        order_tmp = np.argsort(-kept_sims)
        sims_sorted = kept_sims[order_tmp]
        ids_sorted = kept_ids[order_tmp]

        top_n = min(5, sims_sorted.shape[0])
        top_mean = float(sims_sorted[:top_n].mean())
        thresh = top_mean - float(delta_margin)

        rel_keep = sims_sorted >= thresh
        sims_sorted = sims_sorted[rel_keep]
        ids_sorted = ids_sorted[rel_keep]

        if ids_sorted.size == 0:
            # 혹시라도 비는 경우, 절대컷 통과 중 최상위 1개만 반환
            top_idx_only = int(np.argmax(kept_sims))
            ids_sorted = np.array([kept_ids[top_idx_only]])
            sims_sorted = np.array([float(kept_sims[top_idx_only])], dtype="float32")

        # --- (3) 최종 top-k 선택 ---
        k = int(min(top_k, ids_sorted.shape[0]))
        final_ids = ids_sorted[:k]
        final_sims = sims_sorted[:k]

        results = [
            {"post_id": int(pid), "score": float(sc)}
            for pid, sc in zip(final_ids.tolist(), final_sims.tolist())
        ]
        return {"total": int(total_before_filter), "results": results}
