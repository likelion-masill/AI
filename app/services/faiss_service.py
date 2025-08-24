import os
import io
import json
import faiss
import numpy as np
import threading
import tempfile
from typing import Iterable, Tuple, List, Dict, Optional
import logging
log = logging.getLogger("app.faiss")

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
        log.info("[Upsert] 요청 수신 (post_id=%d, embedding_dim=%d)", post_id, len(embedding))
        # 임베딩 차원 검증
        if len(embedding) != self.dim:
            log.error("[Upsert] dim mismatch: got %d, expected %d", len(embedding), self.dim)
            raise ValueError(f"dim mismatch: got {len(embedding)}, expected {self.dim}")

        # 입력 데이터 전처리
        x = np.asarray(embedding, dtype="float32").reshape(1, -1)
        x = self._normalize(x)
        ids = np.asarray([post_id], dtype="int64")

        with self._lock:
            # postId가 이미 존재한다면 제거 후 새로 추가
            self._index.remove_ids(ids)
            self._index.add_with_ids(x, ids)
            log.info("[Upsert] post_id=%d (현재 총 개수=%d)", post_id, self.ntotal)
            if self.autosave:
                self.save_index()
            return {"upserted": 1, "ntotal": self.ntotal}

    def remove(self, post_id: int) -> int:
        log.info("[Remove] 요청 수신 (post_id=%d)", post_id)
        with self._lock:
            n = int(self._index.remove_ids(np.asarray([post_id], dtype="int64")))
            log.info("[Remove] post_id=%d (삭제된 개수=%d, 현재 총 개수=%d)", post_id, n, self.ntotal)
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
            tau_abs: float = 0.22,  # 최소 점수 임계값(코사인)
            delta_margin: float = 0.06,  # 상대 마진(상위 평균 대비 허용 하락폭)
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
        log.info(f"[부분검색] 시작 top_k={top_k}, 절대임계값(tau_abs)={tau_abs:.3f}, "
                 f"상대마진(delta)={delta_margin:.3f}, 정규화={normalize_query}, "
                 f"차원={self.dim}, 후보군 개수={len(candidate_ids)}")

        if not candidate_ids:
            log.info("[부분검색] 후보군이 비어 있음")
            return {"total": 0, "results": []}

        if len(query_embedding) != self.dim:
            log.info("[부분검색] 차원 불일치: 입력=%s, 기대=%s", len(query_embedding), self.dim)
            raise ValueError(f"query dim mismatch: got {len(query_embedding)}, expected {self.dim}")

        if top_k <= 0:
            log.info("[부분검색] top_k가 0 이하 (top_k=%s)", top_k)
            return {"total": 0, "results": []}

        q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        if self.use_cosine and normalize_query:
            q = self._normalize(q)
            log.info("[부분검색] 쿼리 벡터 정규화 완료 (use_cosine=%s)", self.use_cosine)
        q = q.reshape(-1)  # (d,)

        with self._lock:
            if self.ntotal == 0:
                log.info("[부분검색] 인덱스가 비어 있음 (ntotal=0)")
                return {"total": 0, "results": []}
            try:
                id_arr = faiss.vector_to_array(self._index.id_map)  # numpy array of ids
            except AttributeError:
                log.exception("[부분검색] IndexIDMap2.id_map 속성을 사용할 수 없음")
                raise RuntimeError("IndexIDMap2.id_map not available on this build")

            id_set = set(int(x) for x in id_arr)
            log.info("[부분검색] 인덱스 상태: 총 벡터수=%s, idmap 크기=%s", int(self.ntotal), len(id_set))

            vecs, present_ids = [], []
            miss = 0
            log.info("[부분검색] 입력된 후보군 개수=%s", len(candidate_ids))
            for pid in candidate_ids:
                ipid = int(pid)
                if ipid not in id_set:
                    miss += 1
                    continue
                v = self._index.reconstruct(ipid)  # (d,)
                vecs.append(v)
                present_ids.append(ipid)
            log.info("[부분검색] 인덱스에 존재하는 후보=%s, 누락=%s", len(present_ids), miss)

        if not present_ids:
            log.info("[부분검색] 후보군이 인덱스에 전혀 존재하지 않음")
            return {"total": 0, "results": []}

        # 유사도 계산
        M = np.stack(vecs, axis=0).astype("float32")  # (N, d)
        sims = M @ q  # (N,)
        total_before = len(present_ids)
        log.info("[부분검색] 유사도 통계: 최소=%.4f, 중앙값=%.4f, 상위25%%=%.4f, 최대=%.4f",
                 float(np.min(sims)),
                 float(np.percentile(sims, 50)),
                 float(np.percentile(sims, 75)),
                 float(np.max(sims)))

        # (1) 절대 임계값 필터
        keep_abs = sims >= float(tau_abs)
        kept_cnt = int(np.count_nonzero(keep_abs))
        log.info("[부분검색] 절대 임계값 적용 tau_abs=%.3f → 통과 개수=%s", tau_abs, kept_cnt)

        if kept_cnt == 0:
            # 모두 컷 → 상위 1개 관용 통과
            top_idx_all = int(np.argmax(sims))
            kept_ids = np.array([present_ids[top_idx_all]])
            kept_sims = np.array([float(sims[top_idx_all])], dtype="float32")
            log.info("[부분검색] 절대 임계값 통과 없음 → 최상위 1개 관용 통과 id=%s, 점수=%.4f",
                     int(kept_ids[0]), float(kept_sims[0]))
        else:
            kept_ids = np.array(present_ids, dtype=np.int64)[keep_abs]
            kept_sims = sims[keep_abs]

        # (2) 상대 마진 필터 (상위 평균 대비 Δ)
        order_tmp = np.argsort(-kept_sims)
        sims_sorted = kept_sims[order_tmp]
        ids_sorted = kept_ids[order_tmp]

        top_n = min(5, sims_sorted.shape[0])
        top_mean = float(sims_sorted[:top_n].mean())
        thresh = top_mean - float(delta_margin)

        rel_keep = sims_sorted >= thresh
        kept_after_margin = int(np.count_nonzero(rel_keep))
        log.info("[부분검색] 상대 마진 필터 top평균=%.4f, 마진=%.4f, 임계값=%.4f → 통과 개수=%s, "
                 "상위5점수=%s",
                 top_mean, delta_margin, thresh, kept_after_margin,
                 [float(x) for x in sims_sorted[:min(5, sims_sorted.shape[0])]])

        sims_sorted = sims_sorted[rel_keep]
        ids_sorted = ids_sorted[rel_keep]

        if ids_sorted.size == 0:
            # 혹시 비면 절대컷 통과 중 최상위 1개만
            top_idx_only = int(np.argmax(kept_sims))
            ids_sorted = np.array([kept_ids[top_idx_only]])
            sims_sorted = np.array([float(kept_sims[top_idx_only])], dtype="float32")
            log.info("[부분검색] 상대 마진 통과 없음 → 절대 임계값 최상위 1개 선택 id=%s, 점수=%.4f",
                     int(ids_sorted[0]), float(sims_sorted[0]))

        # (3) 최종 top-k 선택
        k = int(min(top_k, ids_sorted.shape[0]))
        final_ids = ids_sorted[:k]
        final_sims = sims_sorted[:k]

        log.info("[부분검색] 최종 선택 k=%s, ids=%s, 점수=%s",
                 k,
                 [int(x) for x in final_ids.tolist()],
                 [float(x) for x in final_sims.tolist()])

        results = [
            {"post_id": int(pid), "score": float(sc)}
            for pid, sc in zip(final_ids.tolist(), final_sims.tolist())
        ]
        return {"total": int(total_before), "results": results}


    def ai_recommend(
            self,
            query_embedding: List[float],
            candidate_ids: List[int],
            top_k: int = 10,
            normalize_query: bool = True,
    ) -> Dict:
        """
        candidate_ids 서브셋 내부에서만 쿼리 임베딩과의 유사도(내적/코사인)로 상위 top_k 랭킹 반환.
        반환 형태:
          {
            "total": <int>,   # present_ids 개수 (top_k 자르기 전)
            "results": [{"post_id": int, "score": float}, ...]
          }
        """
        log.info("[AI Recommend] 요청 수신 (candidate_ids=%d, top_k=%d, normalize=%s)",
                 len(candidate_ids), top_k, normalize_query)

        if not candidate_ids:
            return {"total": 0, "results": []}
        if len(query_embedding) != self.dim:
            raise ValueError(f"query dim mismatch: got {len(query_embedding)}, expected {self.dim}")
        if top_k <= 0:
            return {"total": 0, "results": []}

        q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        if self.use_cosine and normalize_query:
            q = self._normalize(q)
            log.info("[AI Recommend] 쿼리 벡터 정규화 완료")
        q = q.reshape(-1)  # (d,)

        with self._lock:
            if self.ntotal == 0:
                log.info("[AI Recommend] 인덱스 비어 있음 → 결과 0")
                return {"total": 0, "results": []}
            try:
                id_arr = faiss.vector_to_array(self._index.id_map)  # numpy array of ids
                log.info("[AI Recommend] 인덱스에 등록된 id 수: %d", len(id_arr))
            except AttributeError:
                raise RuntimeError("IndexIDMap2.id_map not available on this build")

            id_set = set(int(x) for x in id_arr)

            vecs = []
            present_ids = []
            for pid in candidate_ids:
                pid = int(pid)
                if pid not in id_set:
                    continue
                v = self._index.reconstruct(pid)  # (d,)
                vecs.append(v)
                present_ids.append(pid)

        if not present_ids:
            log.info("[AI Recommend] 후보군 교집합 없음 → 결과 0")
            return {"total": 0, "results": []}

        log.info("[AI Recommend] 실제 검색 가능한 후보군 수: %d", len(present_ids))


        M = np.stack(vecs, axis=0).astype("float32")  # (N, d)
        sims = M @ q  # (N,)

        k = int(min(top_k, sims.shape[0]))
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results = [
            {"post_id": int(present_ids[i]), "score": float(sims[i])}
            for i in top_idx
        ]
        log.info("[AI Recommend] 최종 결과: total=%d, 반환=%d", len(present_ids), len(results))

        return {"total": len(present_ids), "results": results}