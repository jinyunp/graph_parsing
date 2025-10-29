import os, json, time, base64, requests, hashlib, re
from typing import Tuple, Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from schemas import *
from prompts_chart_keywords import SYSTEM_PROMPT, make_user_prompt, make_keywords_only_prompt
from prompts_semantic_summary import SYSTEM_PROMPT_SUMMARY, make_summary_prompt
from config import (
    BACKEND,
    HF_MODEL_ID, HF_DTYPE, HF_DEVICE_MAP, HF_TRUST_REMOTE_CODE, HF_MAX_NEW_TOKENS, HF_USE_FLASH_ATTN, HF_OFFLOAD_FOLDER,
    OLLAMA_HOST, OLLAMA_MODEL,
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL, OPENROUTER_HTTP_REFERER, OPENROUTER_TITLE, OPENROUTER_FORCE_JSON,
    KEYWORDS_MIN, KEYWORDS_MAX, SUMMARY_MIN_SENT, SUMMARY_MAX_SENT,
    DEBUG, DEBUG_TRACE
)

USER_PROMPT = make_user_prompt(KEYWORDS_MIN, KEYWORDS_MAX)
KEYS_ONLY_PROMPT = make_keywords_only_prompt(KEYWORDS_MIN, KEYWORDS_MAX)

def _torch_dtype_from_str(s):
    if s == "auto": return None
    import torch
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(s, None)

def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_open_image(path: str, max_side: int = 2048) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        r = max_side / float(max(w, h))
        img = img.resize((int(w*r), int(h*r)))
    return img

def _strip_code_fences(t: str) -> str:
    s = t.strip()
    if s.startswith("```"):
        s = s.lstrip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip("\n\r ")
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()

def _remove_bom_and_whitespace(t: str) -> str:
    return t.encode("utf-8", "ignore").decode("utf-8").lstrip("\ufeff").strip()

class JsonParseError(Exception):
    def __init__(self, message, raw_text):
        super().__init__(message)
        self.raw_text = raw_text

def _extract_json_object(text: str) -> dict:
    s = _remove_bom_and_whitespace(_strip_code_fences(text))
    if not s:
        raise JsonParseError("empty string after stripping", text)
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    if start == -1:
        raise JsonParseError("no opening brace found", text)
    stack = 0
    end = -1
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0:
                end = i
                break
    if end == -1:
        raise JsonParseError("unbalanced braces", text)
    candidate = s[start:end+1]
    try:
        return json.loads(candidate)
    except Exception as e:
        raise JsonParseError(f"json.loads candidate failed: {e}", text)

def _parse_keywords_only(raw_text: str) -> List[str]:
    s = _remove_bom_and_whitespace(_strip_code_fences(raw_text))
    if not s:
        return []
    try:
        j = json.loads(s)
        if isinstance(j, list) and all(isinstance(x, str) for x in j):
            return j
        if isinstance(j, dict) and isinstance(j.get("key_phrases"), list):
            return [x for x in j["key_phrases"] if isinstance(x, str)]
    except Exception:
        pass
    start = s.find("[")
    if start != -1:
        stack = 0; end = -1
        for i, ch in enumerate(s[start:], start):
            if ch == "[":
                stack += 1
            elif ch == "]":
                stack -= 1
                if stack == 0:
                    end = i; break
        if end != -1:
            try:
                arr = json.loads(s[start:end+1])
                if isinstance(arr, list):
                    return [x for x in arr if isinstance(x, str)]
            except Exception:
                pass
    if "key_phrases" in s:
        m = re.search(r'"key_phrases"\s*:\s*(\[[^\]]*\])', s, re.S)
        if m:
            try:
                arr = json.loads(m.group(1))
                return [x for x in arr if isinstance(x, str)]
            except Exception:
                pass
    return []

def _filter_known_fields(cls, d: dict):
    allowed = set(f.name for f in cls.__dataclass_fields__.values())
    return {k: v for k, v in (d or {}).items() if k in allowed}

# backend helpers
def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _img_b64_data_url(path: str) -> str:
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"

# HF cache
_HF_MODEL = None
_HF_PROCESSOR = None

def _ensure_hf_loaded():
    global _HF_MODEL, _HF_PROCESSOR
    if _HF_MODEL is not None:
        return
    from transformers import AutoConfig, AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration

    dtype = _torch_dtype_from_str(HF_DTYPE)
    attn_kwargs = {"attn_implementation": "flash_attention_2"} if HF_USE_FLASH_ATTN else {}

    cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=HF_TRUST_REMOTE_CODE)
    mt = getattr(cfg, "model_type", "")
    if "qwen2_5_vl" not in mt:
        raise RuntimeError(f"HF_MODEL_ID={HF_MODEL_ID}는 qwen2_5_vl 아키텍처가 아닙니다. (model_type={mt})")

    common_kwargs = dict(
        device_map=HF_DEVICE_MAP,
        torch_dtype=dtype,
        trust_remote_code=HF_TRUST_REMOTE_CODE,
        **attn_kwargs
    )
    if HF_OFFLOAD_FOLDER:
        os.makedirs(HF_OFFLOAD_FOLDER, exist_ok=True)
        common_kwargs["offload_folder"] = HF_OFFLOAD_FOLDER

    _HF_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(HF_MODEL_ID, **common_kwargs)
    _HF_PROCESSOR = AutoProcessor.from_pretrained(HF_MODEL_ID, trust_remote_code=HF_TRUST_REMOTE_CODE)

def _call_hf(image_path: str, sys_prompt: str, user_prompt: str) -> Dict[str, Any]:
    import torch
    _ensure_hf_loaded()
    img = _safe_open_image(image_path)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": user_prompt}]},
    ]
    text = _HF_PROCESSOR.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = _HF_PROCESSOR(text=[text], images=[img], return_tensors="pt").to(_HF_MODEL.device)
    with torch.no_grad():
        out = _HF_MODEL.generate(**inputs, max_new_tokens=HF_MAX_NEW_TOKENS, do_sample=False)
    out_text = _HF_PROCESSOR.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    return {"backend": "hf", "model": HF_MODEL_ID, "message": {"content": out_text}}

def _call_ollama(image_path: str, sys_prompt: str, user_prompt: str) -> Dict[str, Any]:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt, "images": [_img_to_b64(image_path)]},
        ],
        "options": {"temperature": 0}
    }
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()

def _call_openrouter(image_path: str, sys_prompt: str, user_prompt: str, force_json=False) -> Dict[str, Any]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
        "X-Title": OPENROUTER_TITLE,
    }
    content = [
        {"type": "text", "text": user_prompt},
        {"type": "image_url", "image_url": {"url": _img_b64_data_url(image_path)}},
    ]
    payload = {
        "model": OPENROUTER_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content}
        ]
    }
    if force_json:
        payload["response_format"] = {"type": "json_object"}
    r = requests.post(f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions", json=payload, headers=headers, timeout=300)
    r.raise_for_status()
    return r.json()

def _primary_call_json(image_path: str) -> Dict[str, Any]:
    if BACKEND == "ollama":
        return _call_ollama(image_path, SYSTEM_PROMPT, USER_PROMPT)
    elif BACKEND == "openrouter":
        return _call_openrouter(image_path, SYSTEM_PROMPT, USER_PROMPT, force_json=OPENROUTER_FORCE_JSON)
    else:
        return _call_hf(image_path, SYSTEM_PROMPT, USER_PROMPT)

def _keywords_only_call(image_path: str) -> Dict[str, Any]:
    if BACKEND == "ollama":
        return _call_ollama(image_path, SYSTEM_PROMPT, KEYS_ONLY_PROMPT)
    elif BACKEND == "openrouter":
        return _call_openrouter(image_path, SYSTEM_PROMPT, KEYS_ONLY_PROMPT, force_json=False)
    else:
        return _call_hf(image_path, SYSTEM_PROMPT, KEYS_ONLY_PROMPT)

def _summary_call(image_path: str, keywords: List[str]) -> Dict[str, Any]:
    prompt = make_summary_prompt(keywords, SUMMARY_MIN_SENT, SUMMARY_MAX_SENT)
    if BACKEND == "ollama":
        return _call_ollama(image_path, SYSTEM_PROMPT_SUMMARY, prompt)
    elif BACKEND == "openrouter":
        return _call_openrouter(image_path, SYSTEM_PROMPT_SUMMARY, prompt, force_json=False)
    else:
        return _call_hf(image_path, SYSTEM_PROMPT_SUMMARY, prompt)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type((RuntimeError,)))
def infer_chart_metadata_from_image(image_path: str) -> Tuple[ChartMetadata, str, Dict[str, Any], Dict[str, float]]:
    t0 = time.perf_counter()
    raw_http = _primary_call_json(image_path)
    if BACKEND == "ollama":
        raw_text = (raw_http.get("message") or {}).get("content", "") or (raw_http.get("response") or "")
    elif BACKEND == "openrouter":
        raw_text = (raw_http.get("choices") or [{}])[0].get("message", {}).get("content", "")
    else:
        raw_text = (raw_http.get("message") or {}).get("content", "")
    t1 = time.perf_counter()

    try:
        data = _extract_json_object(raw_text)
        parse_failed = False
    except JsonParseError:
        parse_failed = True
        data = {
            "is_chart": False,
            "chart_type": None,
            "orientation": "unknown",
            "title": {"text": None, "is_inferred": False},
            "x_axis": {"name": None, "unit": None, "is_inferred": False, "scale": "unknown"},
            "y_axis": {"name": None, "unit": None, "is_inferred": False, "scale": "unknown"},
            "secondary_y_axis": {"name": None, "unit": None, "is_inferred": False, "scale": "unknown"},
            "legend": {"present": False, "labels": [], "location_hint": None},
            "data_series_count": None,
            "series": [],
            "subplots": [],
            "annotations_present": False,
            "annotations": [],
            "table_like": False,
            "grid_present": None,
            "background_image_present": None,
            "caption_nearby": None,
            "quality_flags": {
                "low_resolution": False,
                "cropped_or_cutoff": False,
                "non_korean_text_present": False,
                "heavy_watermark": False,
                "skew_or_perspective": False
            },
            "confidence": 0.0,
            "source": {"source_pdf": None, "page_number": None, "image_path": None, "image_sha1": None, "bbox": None},
            "key_phrases": [],
            "extra": {"error": "json_parse_error"}
        }

    keywords_retry = False
    if parse_failed:
        try:
            kw_http = _keywords_only_call(image_path)
            if BACKEND == "ollama":
                kw_text = (kw_http.get("message") or {}).get("content", "") or (kw_http.get("response") or "")
            elif BACKEND == "openrouter":
                kw_text = (kw_http.get("choices") or [{}])[0].get("message", {}).get("content", "")
            else:
                kw_text = (kw_http.get("message") or {}).get("content", "")
            kws = _parse_keywords_only(kw_text)
            if kws:
                data["key_phrases"] = kws
            raw_http = {"primary": raw_http, "keywords_only": kw_http}
            raw_text = raw_text + "\n\n---\n[keywords_only]\n" + (kw_text or "")
            keywords_retry = True
        except Exception:
            pass

    data.setdefault("source", {})
    data["source"]["image_path"] = image_path
    data["source"]["image_sha1"] = _file_sha1(image_path)

    title_d = data.get("title") or {}
    meta = ChartMetadata(
        is_chart=bool(data.get("is_chart", False)),
        chart_type=data.get("chart_type"),
        orientation=Orientation(data.get("orientation", "unknown")) if data.get("orientation") in [e.value for e in Orientation] else Orientation.unknown,
        title=TitleField(**_filter_known_fields(TitleField, title_d)),
        x_axis=AxisField(**_filter_known_fields(AxisField, data.get("x_axis"))),
        y_axis=AxisField(**_filter_known_fields(AxisField, data.get("y_axis"))),
        secondary_y_axis=AxisField(**_filter_known_fields(AxisField, data.get("secondary_y_axis"))),
        legend=LegendField(**_filter_known_fields(LegendField, data.get("legend"))),
        data_series_count=data.get("data_series_count"),
        series=[SeriesItem(**_filter_known_fields(SeriesItem, s)) for s in (data.get("series") or [])],
        subplots=[SubplotMeta(**_filter_known_fields(SubplotMeta, sp)) for sp in (data.get("subplots") or [])],
        annotations_present=bool(data.get("annotations_present", False)),
        annotations=list(data.get("annotations") or []),
        table_like=bool(data.get("table_like", False)),
        grid_present=data.get("grid_present"),
        background_image_present=data.get("background_image_present"),
        caption_nearby=data.get("caption_nearby"),
        key_phrases=list(data.get("key_phrases") or []),
        confidence=float(data.get("confidence", 0.0)),
        source=SourceRef(image_path=image_path, image_sha1=data["source"]["image_sha1"]),
    )

    t2 = time.perf_counter()
    timings = {"gen_sec": (t1 - t0), "struct_sec": (t2 - t1), "keywords_retry": keywords_retry}
    return meta, raw_text, raw_http, timings

def generate_semantic_summary(image_path: str, keywords: List[str]) -> tuple[str, Dict[str, Any]]:
    raw = _summary_call(image_path, keywords or [])
    if BACKEND == "ollama":
        text = (raw.get("message") or {}).get("content", "") or (raw.get("response") or "")
    elif BACKEND == "openrouter":
        text = (raw.get("choices") or [{}])[0].get("message", {}).get("content", "")
    else:
        text = (raw.get("message") or {}).get("content", "")
    return (text or "").strip(), raw
