# -*- coding: utf-8 -*-
"""
Step 2: 이미지 + 키워드 기반 의미 요약(semantic summary) 프롬프트
- SYSTEM_PROMPT_SUMMARY: 의미 중심 도메인 요약
- make_summary_prompt(): 사용자 프롬프트
"""
from typing import List

SYSTEM_PROMPT_SUMMARY = (
    "You are a careful vision-language analyst for ironmaking/steelmaking charts. "
    "Given an image of a chart and a set of high-signal Korean keywords (already extracted from that chart), "
    "write a concise Korean semantic summary that captures relationships, trends, thresholds, optima, regimes, "
    "and domain mechanisms. Avoid raw OCR noise, file names, URLs, or page artifacts.\n\n"
    "GUIDELINES:\n"
    "- Focus on SEMANTICS over raw tokens: 관계/추세/임계/최적/메커니즘/영역을 서술\n"
    "- Prefer ironmaking domain terms when suitable: 슬래그 점도, 염기도(Basicity), 유동성, 탈황, 용선, 소결광 등\n"
    "- If uncertainty exists, state it explicitly (e.g., '정확한 축 단위는 불명확')\n"
    "- Keep it compact and readable; do not invent data not supported by the chart/keywords.\n"
)

def make_summary_prompt(keywords: List[str], min_sentences: int = 3, max_sentences: int = 6) -> str:
    kw = [k for k in (keywords or []) if isinstance(k, str)]
    kw_block = "· " + "\n· ".join(kw) if kw else "(키워드 없음)"
    return (
        "아래 그래프 이미지를 참고하고, 제공된 키워드 목록을 근거로 의미 중심의 한국어 요약을 작성해 주세요.\n"
        f"- 분량: {min_sentences}~{max_sentences}문장\n"
        "- 포함: 핵심 관계/추세, 임계점/최적 범위, 도메인 메커니즘(가능시), 한계나 불확실성\n"
        "- 금지: 불필요한 수치 나열, 파일/URL/워터마크, 과도한 추측\n"
        "\n[키워드]\n"
        f"{kw_block}\n"
        "\n출력 형태: 순수 한국어 요약문만 출력 (코드펜스/메타데이터 금지)"
    )
