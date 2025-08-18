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
    SYSTEM = (
        "너는 한국어 이벤트 요약 정리자야. 과장 없이 사실만 간결히, "
        "원문(draft)에 없는 내용은 절대 추정/창작하지 마라."
    )

    USER_TMPL = (
        "다음 드래프트 요약에서 핵심 정보만 뽑아, 오직 '항목: 값' 형식의 줄들로만 출력하라.\n"
        "규칙:\n"
        "1) 각 줄은 반드시 '항목: 값' 형태여야 한다. 문장형 요약, 목록 기호, 번호 금지.\n"
        "2) 원문에 '명시된' 정보만 포함한다. 알 수 없는 항목은 아예 쓰지 않는다(미제공/null/없음 금지).\n"
        "3) 의미가 같은 표현은 다음과 같이 '표준 항목명'으로 정규화한다 (있을 때만):\n"
        "   - 대상/참여대상/이용대상 → 대상\n"
        "   - 일정/일시/운영시간/기간 → 일정\n"
        "   - 장소/위치/주소 → 장소\n"
        "   - 참가비/비용/요금/금액 → 금액\n"
        "   - 혜택/지원/특전/증정 → 혜택\n"
        "   - 문의/연락처/전화/이메일/URL → 문의\n"
        "   - 주최/주관/후원 → 주최\n"
        "   - 프로그램/부스/행사내용 → 프로그램\n"
        "   - 사전신청/예약/접수 → 신청\n"
        "   - 정원/모집인원 → 정원\n"
        "   - 유의사항/준비물/주의 → 유의사항\n"
        "4) 값은 불필요한 수식어 제거 후 간결하게 요지로만 쓴다.\n"
        "5) 중복 항목은 합치고, 항목 수는 3~8줄 내로 압축한다(가능하면).\n"
        "6) 줄 사이 공백 줄 없이 출력한다.\n\n"
        "[드래프트]\n{draft}"
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
