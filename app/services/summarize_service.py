# app/services/summarize_service.py
import os
import re
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import logging
log = logging.getLogger("app.faiss")

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
    log.info("[TextRank] 원본 텍스트 길이: %d", len(text))

    # 한국어 문장 분할
    sents = sent_tokenize_ko(text)
    log.info("[TextRank] 분리된 문장 수: %d", len(sents))

    # 문장 개수가 tok_k 이하인 경우, 그냥 원문 반환
    if len(sents) <= top_k:
        log.info("[TextRank] 문장 수가 %d 이하 → 원문 반환", top_k)
        return " ".join(sents)

    # TF-IDF기반 문장 벡터화
    vec = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1, 2))
    X = vec.fit_transform(sents)
    log.info("[TextRank] TF-IDF 행렬 크기: %s", X.shape)


    # TF-IDF 행렬을 기반으로 코사인 유사도 계산
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)


    # Pagerank 알고리즘을 사용하여 문장 중요도 계산
    G = nx.from_numpy_array(sim)
    scores = nx.pagerank(G)

    # 중요도 점수를 기준으로 문장 정렬 후 상위 top_k개 선택
    ranked = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)
    chosen = sorted([i for i in ranked if len(sents[i]) >= min_len][:top_k])
    log.info("[TextRank][성공] 최종 선택 문장 인덱스: %s", chosen)

    return " ".join(sents[i] for i in chosen)

# ---------- OpenAI 호출 (GPT-4o-mini) ----------
# draft : 1차 추출 요약 결과
# temperature : 생성 요약의 다양성 조절 (0.0 ~ 1.0)
#   - 낮을수록 일관성 있는 요약 생성
# max_tokens : 생성 요약의 최대 토큰 수
def refine_gpt(draft: str, temperature: float = 0.3, max_tokens: int = 300) -> str:
    SYSTEM = (
        "너는 한국어 이벤트 요약 정리자다. "
        "출력은 반드시 key: value 형식으로만 구성해야 한다. "
        "텍스트에 명시된 내용만 사용하고, 추측·창작은 절대 금지한다. "
        "허용된 key 외에는 출력하지 마라."
    )

    USER_TMPL = (
        "아래 텍스트에서 핵심 정보만 key: value 형식으로 정리하라.\n"
        "체크리스트:\n"
        "1) 각 줄은 반드시 'key: value' 형식이어야 한다.\n"
        "2) 텍스트에 명시되지 않은 key는 아예 출력하지 마라. "
        "절대로 '미제공/null/없음/해당 없음' 같은 문구를 쓰지 마라. "
        "값이 없으면 그 key 줄 자체를 출력하지 않는다.\n"
        "3) 값은 간결한 구/한 문장으로 정리한다 (2~80자).\n"
        "4) 요약은 본문 전체 의미를 2~3문장으로 압축한다.\n"
        "5) 중복 key는 출력하지 않는다.\n"
        "6) 일시는 본문에 연도가 있으면 'YYYY.MM.DD(요일)' 형식으로, "
        "연도가 없으면 'MM.DD(요일)' 형식으로 출력한다. "
        "기간일 경우 'MM.DD(요일) ~ MM.DD(요일)' 또는 "
        "'YYYY.MM.DD ~ YYYY.MM.DD' 형식을 사용한다. "
        "월과 일은 반드시 포함해야 하며, 요일이 있으면 함께 표기한다.\n"
        "7) 제목, 일시, 장소는 화면 상단 UI에 이미 표시될 수 있으므로, "
        "요약에서는 반드시 필요하지 않다면 생략한다.\n\n"

        "[허용된 key]\n"
        "- 공통: 제목, 일시, 장소, 주최/주관, 대상, 요약\n"
        "- 확장: 신청방법, 문의, 비용, 정원\n"
        "- 카테고리 특화: 강사, 커리큘럼, 준비물, 모집기간, 활동내용, 활동장소, "
        "판매품목, 참가조건, 프로그램, 공연/전시명, 아티스트, 가게명, 이벤트내용, "
        "공모주제, 공모기간, 응모방법\n\n"
        "[본문]\n{draft}"
    )

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(draft=draft)},
            ],
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        result = resp.output_text.strip()
        log.info("[RefineGPT] 요약 성공 (결과 길이=%d)", len(result))
        return result

    except Exception as e:
        log.error("[RefineGPT] OpenAI 호출 실패: %s", e, exc_info=True)
        raise
