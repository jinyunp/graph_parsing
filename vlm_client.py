import base64, json, hashlib, requests, os, traceback
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from typing import Tuple, Dict, Any
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from schemas import *
from prompts_chart_keywords import SYSTEM_PROMPT, make_user_prompt
from config import (
    BACKEND,
    HF_MODEL_ID, HF_DTYPE, HF_DEVICE_MAP, HF_TRUST_REMOTE_CODE, HF_MAX_NEW_TOKENS, HF_USE_FLASH_ATTN, HF_OFFLOAD_FOLDER,
    OLLAMA_HOST, OLLAMA_MODEL,
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL, OPENROUTER_HTTP_REFERER, OPENROUTER_TITLE, OPENROUTER_FORCE_JSON,
    KEYWORDS_MIN, KEYWORDS_MAX, DEBUG, DEBUG_TRACE
)

USER_PROMPT = make_user_prompt(KEYWORDS_MIN, KEYWORDS_MAX)

# ---------------------- 공통 유틸 ----------------------
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

def _torch_dtype_from_str(s):
    if s == "auto": return None
    import torch
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(s, None)

def _safe_open_image(path: str, max_side: int = 2048) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        r = max_side / float(max(w, h))
        img = img.resize((int(w*r), int(h*r)))
    return img

# ---------------------- Ollama/OpenRouter ----------------------
def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _img_b64_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{path.split('.')[-1]};base64,{b64}"

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
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
        "X-Title": OPENROUTER_TITLE,
    }
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
    if OPENROUTER_FORCE_JSON:
        payload["response_format"] = {"type": "json_object"}
    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    if not resp.ok:
        raise RuntimeError(f"[OpenRouter HTTP 오류] {resp.status_code}\n본문: {resp.text[:2000]}")
    return resp.json()

# ---------------------- HF Transformers (Qwen2.5 대응 강화) ----------------------
_HF_MODEL = None
_HF_PROCESSOR = None

def _ensure_hf_loaded():
    global _HF_MODEL, _HF_PROCESSOR
    if _HF_MODEL is not None:
        return
    from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq
    dtype = _torch_dtype_from_str(HF_DTYPE)
    attn_kwargs = {"attn_implementation": "flash_attention_2"} if HF_USE_FLASH_ATTN else {}

    cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=HF_TRUST_REMOTE_CODE)
    model_type = getattr(cfg, "model_type", "") or ""

    common_kwargs = dict(
        device_map=HF_DEVICE_MAP,
        torch_dtype=dtype,
        trust_remote_code=HF_TRUST_REMOTE_CODE,
        **attn_kwargs
    )
    if HF_OFFLOAD_FOLDER:
        os.makedirs(HF_OFFLOAD_FOLDER, exist_ok=True)
        common_kwargs["offload_folder"] = HF_OFFLOAD_FOLDER

    try:
        if "qwen3_vl" in model_type or "qwen2_5_vl" in model_type:
            # Qwen3-VL / Qwen2.5-VL → CausalLM 경로 우선
            _HF_MODEL = AutoModelForVision2Seq.from_pretrained(HF_MODEL_ID, **common_kwargs)
        elif "qwen2_vl" in model_type:
            # Qwen2-VL 구형
            try:
                from transformers import Qwen2VLForConditionalGeneration
                _HF_MODEL = Qwen2VLForConditionalGeneration.from_pretrained(HF_MODEL_ID, **common_kwargs)
            except Exception:
                _HF_MODEL = AutoModelForVision2Seq.from_pretrained(HF_MODEL_ID, **common_kwargs)
        else:
            # 기타 비전-언어 모델 일반 경로
            _HF_MODEL = AutoModelForVision2Seq.from_pretrained(HF_MODEL_ID, **common_kwargs)
    except OSError as e:
        if DEBUG:
            print("[HF LOAD OSError]", str(e))
        if DEBUG_TRACE:
            traceback.print_exc()
        raise

    _HF_PROCESSOR = AutoProcessor.from_pretrained(HF_MODEL_ID, trust_remote_code=HF_TRUST_REMOTE_CODE)

def _call_hf(image_path: str) -> Dict[str, Any]:
    import torch
    _ensure_hf_loaded()
    try:
        img = _safe_open_image(image_path)
    except OSError as e:
        if DEBUG:
            print("[IMAGE OSError]", str(e))
        if DEBUG_TRACE:
            traceback.print_exc()
        raise

    # 공통 대화 템플릿 (Qwen2.5/Qwen3 모두 호환)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": USER_PROMPT}]},
    ]

    text = _HF_PROCESSOR.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = _HF_PROCESSOR(text=[text], images=[img], return_tensors="pt").to(_HF_MODEL.device)

    try:
        with torch.no_grad():
            out_tokens = _HF_MODEL.generate(**inputs, max_new_tokens=HF_MAX_NEW_TOKENS, do_sample=False)
    except OSError as e:
        if DEBUG:
            print("[GENERATE OSError]", str(e))
        if DEBUG_TRACE:
            traceback.print_exc()
        raise

    out_text = _HF_PROCESSOR.batch_decode(
        out_tokens[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0].strip()

    return {"backend": "hf", "model": HF_MODEL_ID, "message": {"content": out_text}}

# OSError는 재시도하지 않고 즉시 실패
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type((RuntimeError,)))
def infer_chart_metadata_from_image(image_path: str) -> Tuple[ChartMetadata, str, Dict[str, Any]]:
    if BACKEND == "openrouter":
        raw_http = _call_openrouter(image_path)
        raw_text = (raw_http.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    elif BACKEND == "ollama":
        raw_http = _call_ollama(image_path)
        raw_text = (raw_http.get("message") or {}).get("content", "") or ""
    else:
        raw_http = _call_hf(image_path)
        raw_text = (raw_http.get("message") or {}).get("content", "") or ""

    if not raw_text.strip():
        raise RuntimeError("모델 응답 content가 비어있습니다. (모델/백엔드 확인)")

    data = _safe_json_loads(raw_text)
    data.setdefault("source", {})
    data["source"]["image_path"] = image_path
    data["source"]["image_sha1"] = _file_sha1(image_path)

    def _coerce_enum(val, enum_cls, default):
        try:
            if val is None: return default
            return enum_cls(val)
        except: return default
    def _build_axis(d):
        if d is None: return AxisField()
        return AxisField(
            name=d.get("name"), unit=d.get("unit"),
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
        source=SourceRef(image_path=data["source"]["image_path"], image_sha1=data["source"]["image_sha1"]),
    )
    return meta, raw_text, raw_http
