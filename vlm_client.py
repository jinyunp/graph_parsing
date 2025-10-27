import base64
import json
import hashlib
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Tuple, Dict, Any
from schemas import *
from config import (
    BACKEND,
    OLLAMA_HOST, OLLAMA_MODEL,
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL, OPENROUTER_HTTP_REFERER, OPENROUTER_TITLE,
    KEYWORDS_MIN, KEYWORDS_MAX
)

SYSTEM_PROMPT = (
    "You are a meticulous vision-language analyst that extracts structured metadata from technical charts.\n"
    "Return ONLY one JSON object that follows the provided schema. No code fences, no extra text.\n"
    "If the image is not a chart/graph/plot, set \"is_chart\": false and fill unrelated fields with null/[]/false as appropriate.\n"
    "Use Korean for text copied from the image; when you infer text, set \"is_inferred\": true.\n"
)

USER_PROMPT = f"""
아래 이미지를 보고, 다음 스키마를 만족하는 '단일 JSON 객체'만 출력해 주세요.

요구 사항:
- key_phrases는 그래프를 설명/검색에 유용한 **한국어 키워드**로 {KEYWORDS_MIN}~{KEYWORDS_MAX}개를 생성하세요.
- 키워드는 명사/명사구 중심(필요시 간단 형용사 포함)으로, 중복/유의어 반복을 피하세요.
- 예: "고로", "출선 온도", "시간 변화", "환원 반응", "용선", "슬래그", "압력", "유량", "스테이지", "센서 이상"

Schema (keys only, 타입은 참고용):
{{
  "is_chart": bool,
  "chart_type": "line|bar|stacked_bar|histogram|scatter|area|boxplot|violin|heatmap|pie|donut|timeline|other|null",
  "orientation": "vertical|horizontal|unknown",
  "title": {{"text": string|null, "is_inferred": bool}},
  "x_axis": {{"name": string|null, "unit": string|null, "is_inferred": bool, "scale": "linear|log|categorical|time|unknown", "ticks_examples": string[]}},
  "y_axis": {{"name": string|null, "unit": string|null, "is_inferred": bool, "scale": "linear|log|categorical|time|unknown", "ticks_examples": string[]}},
  "legend": {{"present": bool, "labels": string[], "location_hint": string|null}},
  "subplots": [
    {{"title": string|null, "x_axis": {{}}, "y_axis": {{}}, "series": [{{"name": string|null, "style": {{"color_name": string|null, "style_hint": string|null, "marker": "none|circle|square|triangle|diamond|cross|plus|other"}}, "legend_label": string|null}}], "bbox": null}}
  ],
  "annotations_present": bool,
  "annotations": [{{"text": string, "bbox": null}}],
  "data_series_count": number|null,
  "table_like": bool,
  "caption_nearby": string|null,
  "key_phrases": string[],
  "secondary_y_axis": null | {{"name": string|null, "unit": string|null, "is_inferred": bool, "scale": "linear|log|categorical|time|unknown", "ticks_examples": string[]}},
  "grid_present": true|false|null,
  "background_image_present": true|false|null,
  "quality_flags": {{"low_resolution": bool, "cropped_or_cutoff": bool, "non_korean_text_present": bool, "heavy_watermark": bool, "skew_or_perspective": bool}},
  "confidence": number,
  "source": {{"source_pdf": null, "page_number": null, "image_path": string|null, "image_sha1": string|null, "bbox": null}},
  "extra": {{}}
}}

반드시 JSON 객체만 출력하세요. 추가 설명/코드펜스 금지.
"""

def _img_b64_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{path.split('.')[-1]};base64,{b64}"

def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _file_sha1(path: str) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_json_loads(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[len("json"):].strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        last = t.rfind("}")
        if last != -1:
            return json.loads(t[:last+1])
        raise

def _call_ollama(image_path: str) -> Dict[str, Any]:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT, "images": [_img_to_b64(image_path)]},
        ],
        "options": {"temperature": 0}
    }
    resp = requests.post(url, json=payload, timeout=300)
    if not resp.ok:
        raise RuntimeError(f"[Ollama HTTP 오류] {resp.status_code}\n본문: {resp.text[:1500]}")
    return resp.json()

def _call_openrouter(image_path: str) -> Dict[str, Any]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY가 설정되지 않았습니다.")
    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_TITLE:
        headers["X-Title"] = OPENROUTER_TITLE

    payload = {
        "model": OPENROUTER_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": _img_b64_data_url(image_path)}}
            ]}
        ]
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    if not resp.ok:
        raise RuntimeError(f"[OpenRouter HTTP 오류] {resp.status_code}\n본문: {resp.text[:1500]}")
    return resp.json()

from tenacity import retry, stop_after_attempt, wait_fixed
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def infer_chart_metadata_from_image(image_path: str) -> Tuple[ChartMetadata, str, Dict[str, Any]]:
    if BACKEND == "openrouter":
        raw_http = _call_openrouter(image_path)
        raw_text = (raw_http.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    else:
        raw_http = _call_ollama(image_path)
        raw_text = (raw_http.get("message") or {}).get("content", "") or ""

    if not raw_text.strip():
        raise RuntimeError("모델 응답 content가 비어있습니다.")

    data = _safe_json_loads(raw_text)

    data.setdefault("source", {})
    data["source"]["image_path"] = image_path
    data["source"]["image_sha1"] = _file_sha1(image_path)
    data["source"]["source_pdf"] = None
    data["source"]["page_number"] = None

    # dataclass 매핑
    # 축/범례 등은 간결 버전: 필요한 필드만 우선
    def _coerce_enum(val, enum_cls, default):
        try:
            if val is None:
                return default
            return enum_cls(val)
        except Exception:
            return default

    def _build_axis(d):
        if d is None:
            return AxisField()
        return AxisField(
            name=d.get("name"),
            unit=d.get("unit"),
            is_inferred=bool(d.get("is_inferred", False)),
            scale=_coerce_enum(d.get("scale", "unknown"), ScaleType, ScaleType.unknown),
            ticks_examples=list(d.get("ticks_examples", []) or []),
        )

    title_d = data.get("title", {}) or {}
    meta = ChartMetadata(
        is_chart=bool(data.get("is_chart", False)),
        chart_type=_coerce_enum(data.get("chart_type"), ChartType, None),
        orientation=_coerce_enum(data.get("orientation", "unknown"), Orientation, Orientation.unknown),
        title=TitleField(text=title_d.get("text"), is_inferred=bool(title_d.get("is_inferred", False))),
        x_axis=_build_axis(data.get("x_axis")),
        y_axis=_build_axis(data.get("y_axis")),
        legend=LegendField(present=bool((data.get("legend") or {}).get("present", False)),
                           labels=list((data.get("legend") or {}).get("labels", []) or []),
                           location_hint=(data.get("legend") or {}).get("location_hint")),
        key_phrases=list(data.get("key_phrases", []) or []),
        confidence=float(data.get("confidence", 0.0)),
        source=SourceRef(image_path=data["source"]["image_path"],
                         image_sha1=data["source"]["image_sha1"])
    )
    return meta, raw_text, raw_http
