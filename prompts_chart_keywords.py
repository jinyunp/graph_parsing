# -*- coding: utf-8 -*-
"""
프롬프트 스크립트:
- SYSTEM_PROMPT: 그래프 구조 추출 → 그 결과로부터만 키워드(10~15개) 생성
- make_user_prompt(): 키워드 개수와 함께 사용자 프롬프트를 문자열로 생성
- SCHEMA_HINT: 모델이 따라야 할 JSON 스키마 힌트(필드명 고정)
"""

SYSTEM_PROMPT = (
    "You are a meticulous vision-language analyst that extracts structured metadata from technical charts "
    "and then infers Korean search keywords strictly from that metadata.\n"
    "Return ONLY one JSON object that follows the provided schema. No code fences, no extra text.\n\n"
    "CRITICAL RULES\n"
    "1) First, extract the chart structure (axes, title, legend, series, ticks) as-is from the image.\n"
    "   - Use Korean for text copied from the image.\n"
    "   - If you must infer (e.g., missing axis name), set \"is_inferred\": true on that field.\n"
    "   - When text is unreadable/uncertain, set null and \"is_inferred\": false (do NOT guess).\n\n"
    "2) Second, generate key_phrases (10–15 items) ONLY from the extracted JSON fields "
    "   (title, axes, legend labels, series summaries, time range, units, notable trends/anomalies).\n"
    "   - Key phrases must be high-signal Korean 명사/명사구 (필요시 간단 형용사 포함).\n"
    "   - NO raw OCR noise, NO random codes, NO watermarks, NO file names, NO emails/URLs, NO figure numbers.\n"
    "   - Avoid duplicates or near-synonyms; avoid trivial stopwords.\n\n"
    "3) If the image is not a chart/graph/plot, set \"is_chart\": false and fill unrelated fields with null/[]/false.\n\n"
    "ANTI-NOISE FILTER (for key_phrases and copied texts):\n"
    "- Exclude tokens matching any of:\n"
    "  • URLs/emails: /(https?:\\/\\/|www\\.)|\\S+@\\S+/\n"
    "  • Figure/table markers: /(?i)\\b(fig(?:ure)?|table)\\s*\\d+|표\\s*\\d+|그림\\s*\\d+/\n"
    "  • Watermarks/credits/copyright, page headers/footers, DOIs, footnotes.\n"
    "  • Mixed short alnum codes: /\\b[A-Z]{1,4}-\\d{1,4}\\b/, /\\b[A-F0-9]{6,}\\b/\n"
    "  • Standalone numbers without semantic context.\n"
    "- Keep units like %, ℃, km, ms only when tied to an axis/series meaning.\n\n"
    "OUTPUT: A single JSON object following the schema. No explanations."
)

SCHEMA_HINT = """스키마 (필드명 고정):
{
  "is_chart": true,
  "chart_type": "bar|line|scatter|area|histogram|box|heatmap|pie|mixed|other|null",
  "orientation": "horizontal|vertical|mixed|unknown",
  "title": { "text": null, "is_inferred": false },
  "x_axis": { "name": null, "unit": null, "is_inferred": false, "scale": "linear|log|category|datetime|unknown"},
  "y_axis": { "name": null, "unit": null, "is_inferred": false, "scale": "linear|log|category|datetime|unknown"},
  "secondary_y_axis": { "name": null, "unit": null, "is_inferred": false, "scale": "linear|log|category|datetime|unknown", "ticks_examples": [] },
  "legend": { "present": false, "labels": [], "location_hint": null },
  "data_series_count": null,
  "series": [
    { "label": null, "label_is_inferred": false, "sample_points": [], "style_hint": null, "summary": null }
  ],
  "subplots": [],
  "annotations_present": false,
  "annotations": [],
  "table_like": false,
  "grid_present": null,
  "background_image_present": null,
  "caption_nearby": null,
  "quality_flags": {
    "low_resolution": false,
    "cropped_or_cutoff": false,
    "non_korean_text_present": false,
    "heavy_watermark": false,
    "skew_or_perspective": false
  },
  "confidence": 0.0,
  "source": { "source_pdf": null, "page_number": null, "image_path": null, "image_sha1": null, "bbox": null },
  "key_phrases": []
}
"""

def make_user_prompt(keywords_min: int = 10, keywords_max: int = 15) -> str:
    """
    사용자 프롬프트 문자열을 생성합니다.
    - keywords_min/max: key_phrases 생성 개수 가이드
    """
    return (
        "아래 이미지를 보고, 다음 스키마를 만족하는 '단일 JSON 객체'만 출력해 주세요.\n\n"
        "요구 사항:\n"
        f"- 우선 그래프 구조 정보를 정확히 추출하여 스키마에 채우세요(제목/축/범례/시리즈/눈금 등). 만약 확실하지 않다면 null 또는 false를 출력하여 다음 정보 추출로 넘어가기.\n"
        f"- 그 다음, 추출된 구조 정보만을 근거로 key_phrases를 한국어로 {keywords_min}~{keywords_max}개 추론 생성하세요.\n"
        "  • 명사/명사구 중심(필요 시 간단 형용사 포함)\n"
        "  • 그래프 의미에 대해 포함할 수 있는 phrase 생성\n"
        "  • 그래프에 인식되는 값들에 대한 관계를 작성\n"
        "  • 중복/유의어 반복 금지\n"
        "  • 워터마크/파일명/난수형 토큰/URL/이메일/그림·표 번호 등 OCR 잡음 금지\n"
        "- 확신이 없으면 null 또는 빈 배열을 사용하고 \"is_inferred\": false로 두세요(임의 추측 금지).\n"
        "- JSON 외의 추가 문장, 코드펜스 금지.\n\n"
        "형식 엄수:\n"
        "- json key값들은 제사한 것 외로는 절대 출력 금지\n"
        "- key_phrases는 해당 그래프 관련해서만 생성.\n\n"
        f"{SCHEMA_HINT}"
    )

# JSON 파싱이 실패했을 때 곧바로 fallback으로 가지 말고, VLM을 한 번 더 호출해서 key_phrases만 추출하도록 파이프라인 확장

def make_keywords_only_prompt(keywords_min: int = 10, keywords_max: int = 15) -> str:
    """
    JSON 파싱 실패 시, 키워드만 재시도할 때 쓰는 프롬프트.
    - 출력 형식: JSON 배열(List[str]) 또는 {"key_phrases": [...]}
    - 절대 설명/코드펜스 금지
    """
    return (
        "이 이미지를 보고, 그래프의 제목/축/범례/단위/시리즈/눈금에서 의미 있는 검색 키워드만 추려 "
        f"한국어로 {keywords_min}~{keywords_max}개 생성하세요.\n"
        "출력 형식은 다음 중 하나만 허용:\n"
        "1) [\"키워드1\", \"키워드2\", ...]  # JSON 배열\n"
        "2) {\"key_phrases\": [\"키워드1\", \"키워드2\", ...]}\n"
        "설명/문장/코드펜스 금지."
    )
