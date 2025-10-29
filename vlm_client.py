import json, os, traceback, hashlib
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from typing import Tuple, Dict, Any
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from schemas import *
from config import (
    HF_MODEL_ID, HF_DTYPE, HF_DEVICE_MAP, HF_TRUST_REMOTE_CODE, HF_MAX_NEW_TOKENS, HF_USE_FLASH_ATTN, HF_OFFLOAD_FOLDER,
    KEYWORDS_MIN, KEYWORDS_MAX, DEBUG, DEBUG_TRACE
)
from prompts_chart_keywords import SYSTEM_PROMPT, make_user_prompt, make_keywords_only_prompt
# 응답 생성 및 구조화 작업 시간 표출
import time
import json, re

class JsonParseError(Exception):
    def __init__(self, message, raw_text):
        super().__init__(message)
        self.raw_text = raw_text

def _strip_code_fences(t: str) -> str:
    s = t.strip()
    if s.startswith("```"):
        # ```json ... ``` or ``` ... ```
        s = s.lstrip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip("\n\r ")
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()

def _remove_bom_and_whitespace(t: str) -> str:
    # UTF-8 BOM 제거 + 앞뒤 공백 제거
    return t.encode("utf-8", "ignore").decode("utf-8").lstrip("\ufeff").strip()

def _extract_json_object(text: str) -> dict:
    """
    원문에서 가장 바깥 JSON 객체를 안전하게 추출해 파싱.
    - 코드펜스/앞뒤 설명 제거
    - 첫 '{' ~ 균형 잡힌 '}' 까지 스캔
    """
    s = _remove_bom_and_whitespace(_strip_code_fences(text))
    if not s:
        raise JsonParseError("empty string after stripping", text)

    # 1) 직행 시도
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) 중괄호 균형 스캔
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

def _safe_load_json_or_fallback(raw_text: str) -> dict:
    """
    최대한 파싱하고, 실패하면 폴백 스키마(에러 사유 포함)를 반환.
    """
    try:
        return _extract_json_object(raw_text)
    except JsonParseError as e:
        print("    **** printing fallback")
        return {
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
            "extra": {"error": "json_parse_error", "reason": str(e)}
        }


USER_PROMPT = make_user_prompt(KEYWORDS_MIN, KEYWORDS_MAX)
KEYS_ONLY_PROMPT = make_keywords_only_prompt(KEYWORDS_MIN, KEYWORDS_MAX)

_HF_MODEL = None
_HF_PROCESSOR = "cuda" #None

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

def _ensure_hf_loaded():
    global _HF_MODEL, _HF_PROCESSOR
    if _HF_MODEL is not None:
        return

    from transformers import AutoConfig, AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForVision2Seq

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

def _call_hf(image_path: str) -> Dict[str, Any]:
    import torch
    _ensure_hf_loaded()
    img = _safe_open_image(image_path)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": USER_PROMPT}]},
    ]
    text = _HF_PROCESSOR.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = _HF_PROCESSOR(text=[text], images=[img], return_tensors="pt").to(_HF_MODEL.device)

    with torch.no_grad():
        out_tokens = _HF_MODEL.generate(**inputs, max_new_tokens=HF_MAX_NEW_TOKENS, do_sample=False)

    out_text = _HF_PROCESSOR.batch_decode(
        out_tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()

    return {"backend": "hf", "model": HF_MODEL_ID, "message": {"content": out_text}}

# fallback 이후 keyword 추출만 제시도
def _parse_keywords_only(raw_text: str) -> list[str]:
    s = _remove_bom_and_whitespace(_strip_code_fences(raw_text))
    if not s:
        return []
    # 1) 바로 파싱 시도
    try:
        j = json.loads(s)
        if isinstance(j, list) and all(isinstance(x, str) for x in j):
            return j
        if isinstance(j, dict) and isinstance(j.get("key_phrases"), list):
            return [x for x in j["key_phrases"] if isinstance(x, str)]
    except Exception:
        pass
    # 2) 대괄호 균형 스캔으로 리스트 추출
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
    # 3) 마지막 시도: {"key_phrases": [...]} 조각 추출
    if "key_phrases" in s:
        m = re.search(r'"key_phrases"\s*:\s*(\[[^\]]*\])', s, re.S)
        if m:
            try:
                arr = json.loads(m.group(1))
                return [x for x in arr if isinstance(x, str)]
            except Exception:
                pass
    return []

def _call_hf_keywords_only(image: Image.Image) -> dict:
    """
    동일 모델/프로세서로 키워드만 요청.
    """
    print("    **** keywords만 추출 시도")
    import torch
    _ensure_hf_loaded()
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": KEYS_ONLY_PROMPT}]},
    ]
    text = _HF_PROCESSOR.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = _HF_PROCESSOR(text=[text], images=[image], return_tensors="pt").to(_HF_MODEL.device)
    with torch.no_grad():
        out_tokens = _HF_MODEL.generate(**inputs, max_new_tokens=HF_MAX_NEW_TOKENS, do_sample=False)
    out_text = _HF_PROCESSOR.batch_decode(
        out_tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()
    return {"backend": "hf", "model": HF_MODEL_ID, "message": {"content": out_text}}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type((RuntimeError,)))
def infer_chart_metadata_from_image(image_path: str):
    """
    반환: (meta, raw_text, raw_http, timings)
    timings = {"gen_sec": float, "struct_sec": float, "keywords_retry": bool}
    """
    t0 = time.perf_counter()
    raw_http = _call_hf(image_path)                 # 1차: 전체 JSON 생성
    raw_text = (raw_http.get("message") or {}).get("content", "") or ""
    t1 = time.perf_counter()

    data = _safe_load_json_or_fallback(raw_text)
    parse_failed = bool(data.get("extra") == {"error": "json_parse_error"} or data.get("extra", {}).get("error") == "json_parse_error")

    keywords_retry = False
    if parse_failed:
        # 파싱 실패 시: 키워드만 1회 재시도
        try:
            img = _safe_open_image(image_path)
            kw_http = _call_hf_keywords_only(img)
            kw_text = (kw_http.get("message") or {}).get("content", "") or ""
            keywords = _parse_keywords_only(kw_text)
            if keywords:
                data["key_phrases"] = keywords
            # raw 묶음 저장을 위해 합치기
            raw_http = {"primary": raw_http, "keywords_only": kw_http}
            raw_text = raw_text + "\n\n---\n[keywords_only]\n" + kw_text
            keywords_retry = True
        except Exception:
            pass

    # ---------- dataclass 매핑 ----------
    def _coerce_enum(val, enum_cls, default):
        try:
            if val is None: return default
            return enum_cls(val)
        except: return default

    # 안전 필터(정의된 키만 유지) — optional
    def _filter_known_fields(cls, d: dict):
        allowed = set(f.name for f in cls.__dataclass_fields__.values())
        return {k: v for k, v in (d or {}).items() if k in allowed}

    title_d = data.get("title") or {}
    meta = ChartMetadata(
        is_chart=bool(data.get("is_chart", False)),
        chart_type=data.get("chart_type"),
        orientation=_coerce_enum(data.get("orientation", "unknown"), Orientation, Orientation.unknown),
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
        source=SourceRef(image_path=image_path, image_sha1=_file_sha1(image_path)),
    )

    t2 = time.perf_counter()
    timings = {"gen_sec": (t1 - t0), "struct_sec": (t2 - t1), "keywords_retry": keywords_retry}
    return meta, raw_text, raw_http, timings