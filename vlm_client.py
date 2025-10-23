import base64
import json
import hashlib
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Optional
from schemas import *
from config import OLLAMA_HOST, VLM_MODEL

SYSTEM_PROMPT = (
    "You are a meticulous vision-language analyst that extracts structured metadata from technical charts.\n"
    "Return ONLY one JSON object that follows the provided schema. No code fences, no extra text.\n"
    "If the image is not a chart/graph/plot, set \"is_chart\": false and fill unrelated fields with null/[]/false as appropriate.\n"
    "Use Korean for text copied from the image; when you infer text, set \"is_inferred\": true.\n"
)

USER_PROMPT = """
아래 이미지를 보고, 다음 스키마를 만족하는 '단일 JSON 객체'만 출력해 주세요.
(필드가 확인되지 않으면 null/false/[]로 채우세요)
"""

def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _file_sha1(path: str) -> str:
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

def json_to_dataclass(data: dict) -> ChartMetadata:
    is_chart = bool(data.get("is_chart", False))
    chart_type = _coerce_enum(data.get("chart_type"), ChartType, None)
    orientation = _coerce_enum(data.get("orientation", "unknown"), Orientation, Orientation.unknown)

    title_d = data.get("title", {}) or {}
    title = TitleField(text=title_d.get("text"), is_inferred=bool(title_d.get("is_inferred", False)))

    x_axis = _build_axis(data.get("x_axis"))
    y_axis = _build_axis(data.get("y_axis"))

    lg = data.get("legend", {}) or {}
    legend = LegendField(
        present=bool(lg.get("present", False)),
        labels=list(lg.get("labels", []) or []),
        location_hint=lg.get("location_hint")
    )

    subplots = []
    for sp in data.get("subplots", []) or []:
        series_items = []
        for se in sp.get("series", []) or []:
            style = se.get("style") or {}
            series_items.append(
                DataSeries(
                    name=se.get("name"),
                    style=SeriesStyle(
                        color_name=style.get("color_name"),
                        style_hint=style.get("style_hint"),
                        marker=_coerce_enum(style.get("marker", "none"), MarkerType, MarkerType.none)
                    ),
                    legend_label=se.get("legend_label")
                )
            )
        subplots.append(
            SubplotMeta(
                title=sp.get("title"),
                x_axis=_build_axis(sp.get("x_axis")),
                y_axis=_build_axis(sp.get("y_axis")),
                series=series_items,
                bbox=None
            )
        )

    annotations = []
    for a in data.get("annotations", []) or []:
        txt = a.get("text")
        if txt:
            annotations.append(AnnotationItem(text=txt, bbox=None))

    secondary_y_axis = _build_axis(data.get("secondary_y_axis")) if isinstance(data.get("secondary_y_axis"), dict) else None

    qf = data.get("quality_flags", {}) or {}
    quality_flags = QualityFlags(
        low_resolution=bool(qf.get("low_resolution", False)),
        cropped_or_cutoff=bool(qf.get("cropped_or_cutoff", False)),
        non_korean_text_present=bool(qf.get("non_korean_text_present", False)),
        heavy_watermark=bool(qf.get("heavy_watermark", False)),
        skew_or_perspective=bool(qf.get("skew_or_perspective", False)),
    )

    src = data.get("source", {}) or {}
    source = SourceRef(
        source_pdf=None,
        page_number=None,
        image_path=src.get("image_path"),
        image_sha1=src.get("image_sha1"),
        bbox=None
    )

    return ChartMetadata(
        is_chart=is_chart,
        chart_type=chart_type,
        orientation=orientation,
        title=title,
        x_axis=x_axis,
        y_axis=y_axis,
        legend=legend,
        subplots=subplots,
        annotations_present=bool(data.get("annotations_present", False)),
        annotations=annotations,
        data_series_count=data.get("data_series_count"),
        table_like=bool(data.get("table_like", False)),
        caption_nearby=data.get("caption_nearby"),
        key_phrases=list(data.get("key_phrases", []) or []),
        secondary_y_axis=secondary_y_axis,
        grid_present=data.get("grid_present"),
        background_image_present=data.get("background_image_present"),
        quality_flags=quality_flags,
        confidence=float(data.get("confidence", 0.0)),
        source=source,
        extra=dict(data.get("extra", {}) or {})
    )

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def infer_chart_metadata_from_image(image_path: str) -> ChartMetadata:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"

    payload = {
        "model": VLM_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT, "images": [_img_to_b64(image_path)]},
        ],
        "options": {"temperature": 0}
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
    except requests.exceptions.ConnectionError as ce:
        raise RuntimeError(f"[연결 실패] OLLAMA_HOST='{OLLAMA_HOST}' 접속 오류. Ollama 실행/포트 확인: {ce}") from ce
    except requests.exceptions.Timeout as te:
        raise RuntimeError(f"[타임아웃] Ollama 응답 지연. 모델 로딩/큰 이미지 가능성: {te}") from te

    if not resp.ok:
        raise RuntimeError(
            f"[Ollama HTTP 오류] {resp.status_code}\n"
            f"요청 URL: {url}\n"
            f"모델: {VLM_MODEL}\n"
            f"응답 본문(앞부분): {resp.text[:1500]}"
        )

    data_json = resp.json()
    content = (data_json.get("message") or {}).get("content", "")
    if not content.strip():
        raise RuntimeError("[파싱 오류] Ollama 응답에 content 없음. 프롬프트/페이로드 확인.\n"
                           f"원본 응답: {json.dumps(data_json)[:1500]}")

    data = _safe_json_loads(content)
    data.setdefault("source", {})
    data["source"]["image_path"] = image_path
    data["source"]["image_sha1"] = _file_sha1(image_path)
    data["source"]["source_pdf"] = None
    data["source"]["page_number"] = None

    return json_to_dataclass(data)
