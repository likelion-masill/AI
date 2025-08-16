# app/services/summarizer.py
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
def textrank_tfidf(text: str, top_k: int = 5, min_len: int = 10) -> str:
    sents = sent_tokenize_ko(text)
    if len(sents) <= top_k:
        return " ".join(sents)

    vec = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1, 2))
    X = vec.fit_transform(sents)
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)

    G = nx.from_numpy_array(sim)
    scores = nx.pagerank(G)
    ranked = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)
    chosen = sorted([i for i in ranked if len(sents[i]) >= min_len][:top_k])

    return " ".join(sents[i] for i in chosen)

# ---------- OpenAI 호출 (GPT-4o-mini) ----------
def refine_gpt(draft: str, temperature: float = 0.3, max_tokens: int = 300) -> str:
    SYSTEM = "너는 한국어 요약 도우미야. 과장 없이 사실만 간결히 유지해."
    USER_TMPL = (
        "아래 1차 추출 요약을 기반으로, 반드시 '항목: 값' 형식으로만 요약해라.\n"
        "출력 규칙:\n"
        "1. 각 줄은 반드시 '항목명: 값' 형태여야 한다.\n"
        "2. 문장형 요약은 절대 쓰지 마라.\n"
        "3. 불필요한 설명은 절대 하지 않는다.\n\n"
        "[1차 추출 요약]\n{draft}"
    )

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(draft=draft)},
        ],
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return resp.output_text.strip()
