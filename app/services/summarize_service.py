# app/services/summarize_service.py
import os
import re
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ---------- OpenAI 클라이언트 ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- 한국어 문장 분할 ----------
SENT_ENDINGS = r"[\.!\?…]|다|요|죠"
SPLIT_RE = re.compile(rf"(?<={SENT_ENDINGS})\s+")

def sent_tokenize_ko(text: str):
    return [s.strip() for s in SPLIT_RE.split(text) if s.strip()]

# ---------- TextRank + TF-IDF ----------
# text : 요약할 원본 텍스트
# top_k : 추출 요약 시 뽑을 문장 개수
# min_len : 요약 문장의 최소 길이
def textrank_tfidf(text: str, top_k: int = 5, min_len: int = 10) -> str:
    # 한국어 문장 분할
    sents = sent_tokenize_ko(text)
    # 문장 개수가 tok_k 이하인 경우, 그냥 원문 반환
    if len(sents) <= top_k:
        return " ".join(sents)

    # TF-IDF기반 문장 벡터화
    vec = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1, 2))
    X = vec.fit_transform(sents)

    # TF-IDF 행렬을 기반으로 코사인 유사도 계산
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)

    # Pagerank 알고리즘을 사용하여 문장 중요도 계산
    G = nx.from_numpy_array(sim)
    scores = nx.pagerank(G)
    # 중요도 점수를 기준으로 문장 정렬 후 상위 top_k개 선택
    ranked = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)
    chosen = sorted([i for i in ranked if len(sents[i]) >= min_len][:top_k])

    return " ".join(sents[i] for i in chosen)

# ---------- OpenAI 호출 (GPT-4o-mini) ----------
# draft : 1차 추출 요약 결과
# temperature : 생성 요약의 다양성 조절 (0.0 ~ 1.0)
#   - 낮을수록 일관성 있는 요약 생성
# max_tokens : 생성 요약의 최대 토큰 수
def refine_gpt(draft: str, temperature: float = 0.3, max_tokens: int = 300) -> str:
    SYSTEM = "너는 한국어 요약 도우미야. 과장 없이 사실만 간결히 유지해."
    USER_TMPL = (
        "아래 1차 추출 요약을 기반으로, 반드시 '항목: 값' 형식으로만 출력하라.\n"
        "출력 규칙:\n"
        "1. 각 줄은 반드시 '항목명: 값' 형태여야 한다.\n"
        "2. 문장형 요약은 절대 쓰지 마라.\n"
        "3. 불필요한 설명, 중복 항목은 모두 제거하라.\n"
        "4. 반드시 핵심 항목(대상, 기간, 금액, 지원, 문의)만 남겨라.\n\n"
        "[1차 추출 요약]\n{draft}"
    )

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            # 모델의 행동 지침을 지정
            {"role": "system", "content": SYSTEM},
            # 실제 사용자가 하는 요청
            {"role": "user", "content": USER_TMPL.format(draft=draft)},
        ],
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return resp.output_text.strip()
